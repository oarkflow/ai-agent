package content

import (
	"encoding/base64"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

//
// =======================
// Types
// =======================
//

type MediaType string

// MediaType represents MIME types for content.
const (
	// Image
	MediaJPEG MediaType = "image/jpeg"
	MediaPNG  MediaType = "image/png"
	MediaGIF  MediaType = "image/gif"
	MediaWebP MediaType = "image/webp"
	MediaSVG  MediaType = "image/svg+xml"

	// Audio
	MediaMP3  MediaType = "audio/mpeg"
	MediaWAV  MediaType = "audio/wav"
	MediaOGG  MediaType = "audio/ogg"
	MediaFLAC MediaType = "audio/flac"
	MediaM4A  MediaType = "audio/mp4"
	MediaWebM MediaType = "audio/webm"

	// Video
	MediaMP4   MediaType = "video/mp4"
	MediaMOV   MediaType = "video/quicktime"
	MediaAVI   MediaType = "video/x-msvideo"
	MediaMKV   MediaType = "video/x-matroska"
	MediaWebMV MediaType = "video/webm"

	// Documents
	MediaPDF  MediaType = "application/pdf"
	MediaDOCX MediaType = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
	MediaXLSX MediaType = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
	MediaPPTX MediaType = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
	MediaJSON MediaType = "application/json"
	MediaXML  MediaType = "application/xml"
	MediaCSV  MediaType = "text/csv"
	MediaHTML MediaType = "text/html"
	MediaMarkdown MediaType = "text/markdown"
	MediaPlain    MediaType = "text/plain"

	// Code
	MediaGo         MediaType = "text/x-go"
	MediaPython     MediaType = "text/x-python"
	MediaJavaScript MediaType = "text/javascript"
	MediaTypeScript MediaType = "text/typescript"
	MediaJava       MediaType = "text/x-java"
	MediaRust       MediaType = "text/x-rust"
)

type ContentType string

// ContentType represents the type of content.
const (
	TypeText     ContentType = "text"
	TypeImage    ContentType = "image"
	TypeAudio    ContentType = "audio"
	TypeVideo    ContentType = "video"
	TypeDocument ContentType = "document"
	TypeCode     ContentType = "code"
	TypeURL      ContentType = "url"
	TypeFile     ContentType = "file"
)

//
// =======================
// Core Structs
// =======================
//

type Annotation struct {
	Type       string         `json:"type"` // bounding_box, timestamp, text_range
	Label      string         `json:"label,omitempty"`
	Confidence float64        `json:"confidence,omitempty"`
	Data       map[string]any `json:"data,omitempty"`
}

type Content struct {
	Type        ContentType        `json:"type"`
	MediaType   MediaType          `json:"media_type,omitempty"`
	Text        string             `json:"text,omitempty"`
	Data        []byte             `json:"-"`
	Base64Data  string             `json:"data,omitempty"`
	URL         string             `json:"url,omitempty"`
	FilePath    string             `json:"file_path,omitempty"`
	FileName    string             `json:"file_name,omitempty"`
	FileSize    int64              `json:"file_size,omitempty"`
	Metadata    map[string]any     `json:"metadata,omitempty"`
	Language    string             `json:"language,omitempty"`
	Annotations []Annotation       `json:"annotations,omitempty"`
}

//
// =======================
// Helpers
// =======================
//

func getCodeMediaType(language string) MediaType {
	switch strings.ToLower(language) {
	case "go", "golang":
		return MediaGo
	case "python", "py":
		return MediaPython
	case "javascript", "js":
		return MediaJavaScript
	case "typescript", "ts":
		return MediaTypeScript
	case "java":
		return MediaJava
	case "rust", "rs":
		return MediaRust
	default:
		return MediaPlain
	}
}

func detectMediaType(filePath string, data []byte) MediaType {
	ext := strings.ToLower(filepath.Ext(filePath))
	if ext != "" {
		if mt := mime.TypeByExtension(ext); mt != "" {
			return MediaType(strings.Split(mt, ";")[0])
		}
	}

	if len(data) > 0 {
		return MediaType(strings.Split(http.DetectContentType(data), ";")[0])
	}

	return MediaPlain
}

func ContentTypeFromMediaType(mt MediaType) ContentType {
	s := string(mt)

	switch {
	case strings.HasPrefix(s, "image/"):
		return TypeImage
	case strings.HasPrefix(s, "audio/"):
		return TypeAudio
	case strings.HasPrefix(s, "video/"):
		return TypeVideo
	case s == string(MediaPDF),
		strings.Contains(s, "document"),
		strings.Contains(s, "spreadsheet"),
		strings.Contains(s, "presentation"):
		return TypeDocument
	case strings.HasPrefix(s, "text/x-"),
		strings.Contains(s, "javascript"):
		return TypeCode
	case strings.HasPrefix(s, "text/"):
		return TypeText
	default:
		return TypeFile
	}
}

//
// =======================
// Content Methods
// =======================
//

func (c *Content) GetBase64() string {
	if c.Base64Data != "" {
		return c.Base64Data
	}
	if len(c.Data) > 0 {
		c.Base64Data = base64.StdEncoding.EncodeToString(c.Data)
	}
	return c.Base64Data
}

func (c *Content) GetDataURI() string {
	return fmt.Sprintf("data:%s;base64,%s", c.MediaType, c.GetBase64())
}

func (c *Content) IsMultimodal() bool {
	return c.Type == TypeImage || c.Type == TypeAudio || c.Type == TypeVideo
}

func (c *Content) RequiresTranscription() bool {
	return c.Type == TypeAudio || c.Type == TypeVideo
}

func (c *Content) LoadFromReader(r io.Reader) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return fmt.Errorf("failed to read content: %w", err)
	}
	c.Data = data
	c.Base64Data = base64.StdEncoding.EncodeToString(data)
	c.FileSize = int64(len(data))
	return nil
}

//
// =======================
// Constructors
// =======================
//

func NewTextContent(text string) *Content {
	return &Content{
		Type:      TypeText,
		MediaType: MediaPlain,
		Text:      text,
	}
}

func NewCodeContent(code, language string) *Content {
	return &Content{
		Type:      TypeCode,
		Language:  language,
		Text:      code,
		MediaType: getCodeMediaType(language),
	}
}

func NewImageFromFile(filePath string) (*Content, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	mt := detectMediaType(filePath, data)

	return &Content{
		Type:       TypeImage,
		MediaType:  mt,
		Data:       data,
		Base64Data: base64.StdEncoding.EncodeToString(data),
		FilePath:   filePath,
		FileName:   filepath.Base(filePath),
		FileSize:   int64(len(data)),
	}, nil
}

func NewImageFromURL(url string) *Content {
	return &Content{
		Type: TypeImage,
		URL:  url,
	}
}

func NewAudioFromFile(filePath string) (*Content, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	mt := detectMediaType(filePath, data)

	return &Content{
		Type:       TypeAudio,
		MediaType:  mt,
		Data:       data,
		Base64Data: base64.StdEncoding.EncodeToString(data),
		FilePath:   filePath,
		FileName:   filepath.Base(filePath),
		FileSize:   int64(len(data)),
	}, nil
}

func NewVideoFromFile(filePath string) (*Content, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	mt := detectMediaType(filePath, data)

	return &Content{
		Type:       TypeVideo,
		MediaType:  mt,
		Data:       data,
		Base64Data: base64.StdEncoding.EncodeToString(data),
		FilePath:   filePath,
		FileName:   filepath.Base(filePath),
		FileSize:   int64(len(data)),
	}, nil
}

func NewDocumentFromFile(filePath string) (*Content, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	mt := detectMediaType(filePath, data)

	return &Content{
		Type:       TypeDocument,
		MediaType:  mt,
		Data:       data,
		Base64Data: base64.StdEncoding.EncodeToString(data),
		FilePath:   filePath,
		FileName:   filepath.Base(filePath),
		FileSize:   int64(len(data)),
	}, nil
}

func NewURLContent(url string) *Content {
	return &Content{
		Type: TypeURL,
		URL:  url,
	}
}
