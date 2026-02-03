package content

import "time"

//
// =======================
// Roles
// =======================
//

type Role string

// Role represents the role of a message sender.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
	RoleFunction  Role = "function"
)

//
// =======================
// Tool / Function Calls
// =======================
//

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"` // "function"
	Function FunctionCall `json:"function"`
}

//
// =======================
// Message
// =======================
//

type Message struct {
	ID           string         `json:"id,omitempty"`
	Role         Role           `json:"role"`
	Contents     []*Content     `json:"contents"`
	Name         string         `json:"name,omitempty"`
	ToolCallID   string         `json:"tool_call_id,omitempty"`
	FunctionCall *FunctionCall  `json:"function_call,omitempty"`
	ToolCalls    []ToolCall     `json:"tool_calls,omitempty"`
	Timestamp    time.Time      `json:"timestamp,omitempty"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

//
// =======================
// Message Constructors
// =======================
//

func NewTextMessage(role Role, text string) *Message {
	return &Message{
		Role:      role,
		Timestamp: time.Now(),
		Contents: []*Content{
			NewTextContent(text),
		},
	}
}

func NewSystemMessage(text string) *Message {
	return NewTextMessage(RoleSystem, text)
}

func NewUserMessage(text string) *Message {
	return NewTextMessage(RoleUser, text)
}

func NewAssistantMessage(text string) *Message {
	return NewTextMessage(RoleAssistant, text)
}

func NewMultimodalMessage(role Role, contents ...*Content) *Message {
	return &Message{
		Role:      role,
		Contents:  contents,
		Timestamp: time.Now(),
	}
}

//
// =======================
// Message Mutators
// =======================
//

func (m *Message) AddContent(c *Content) *Message {
	m.Contents = append(m.Contents, c)
	return m
}

func (m *Message) AddText(text string) *Message {
	return m.AddContent(NewTextContent(text))
}

func (m *Message) AddCode(code, language string) *Message {
	return m.AddContent(NewCodeContent(code, language))
}

func (m *Message) AddImageFromURL(url string) *Message {
	return m.AddContent(NewImageFromURL(url))
}

func (m *Message) AddImageFromFile(filePath string) (*Message, error) {
	img, err := NewImageFromFile(filePath)
	if err != nil {
		return m, err
	}
	return m.AddContent(img), nil
}

func (m *Message) AddAudioFromFile(filePath string) (*Message, error) {
	audio, err := NewAudioFromFile(filePath)
	if err != nil {
		return m, err
	}
	return m.AddContent(audio), nil
}

func (m *Message) AddVideoFromFile(filePath string) (*Message, error) {
	video, err := NewVideoFromFile(filePath)
	if err != nil {
		return m, err
	}
	return m.AddContent(video), nil
}

func (m *Message) AddDocumentFromFile(filePath string) (*Message, error) {
	doc, err := NewDocumentFromFile(filePath)
	if err != nil {
		return m, err
	}
	return m.AddContent(doc), nil
}

//
// =======================
// Message Queries
// =======================
//

func (m *Message) GetText() string {
	var result string
	for _, c := range m.Contents {
		if c.Type == TypeText || c.Type == TypeCode {
			if result != "" {
				result += "\n"
			}
			result += c.Text
		}
	}
	return result
}

func (m *Message) GetContentTypes() []ContentType {
	set := make(map[ContentType]bool)
	for _, c := range m.Contents {
		set[c.Type] = true
	}

	result := make([]ContentType, 0, len(set))
	for t := range set {
		result = append(result, t)
	}
	return result
}

func (m *Message) HasContent(t ContentType) bool {
	for _, c := range m.Contents {
		if c.Type == t {
			return true
		}
	}
	return false
}

func (m *Message) IsMultimodal() bool {
	for _, c := range m.Contents {
		if c.IsMultimodal() {
			return true
		}
	}
	return false
}

func (m *Message) GetContentByType(t ContentType) []*Content {
	var result []*Content
	for _, c := range m.Contents {
		if c.Type == t {
			result = append(result, c)
		}
	}
	return result
}

func (m *Message) RequiresPreprocessing() bool {
	for _, c := range m.Contents {
		if c.RequiresTranscription() || c.Type == TypeDocument || c.Type == TypeURL {
			return true
		}
	}
	return false
}

//
// =======================
// Message Clone
// =======================
//

func (m *Message) Clone() *Message {
	clone := &Message{
		ID:         m.ID,
		Role:       m.Role,
		Name:       m.Name,
		ToolCallID: m.ToolCallID,
		Timestamp:  m.Timestamp,
	}

	if m.Contents != nil {
		clone.Contents = make([]*Content, len(m.Contents))
		for i, c := range m.Contents {
			cc := *c
			clone.Contents[i] = &cc
		}
	}

	if m.FunctionCall != nil {
		fc := *m.FunctionCall
		clone.FunctionCall = &fc
	}

	if m.ToolCalls != nil {
		clone.ToolCalls = make([]ToolCall, len(m.ToolCalls))
		copy(clone.ToolCalls, m.ToolCalls)
	}

	if m.Metadata != nil {
		clone.Metadata = make(map[string]any)
		for k, v := range m.Metadata {
			clone.Metadata[k] = v
		}
	}

	return clone
}

//
// =======================
// Conversation
// =======================
//

type Conversation struct {
	ID        string         `json:"id"`
	Messages  []*Message     `json:"messages"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
}

func NewConversation(id string) *Conversation {
	now := time.Now()
	return &Conversation{
		ID:        id,
		Messages:  make([]*Message, 0),
		Metadata:  make(map[string]any),
		CreatedAt: now,
		UpdatedAt: now,
	}
}

func (c *Conversation) AddMessage(m *Message) {
	c.Messages = append(c.Messages, m)
	c.UpdatedAt = time.Now()
}

func (c *Conversation) GetLastMessage() *Message {
	if len(c.Messages) == 0 {
		return nil
	}
	return c.Messages[len(c.Messages)-1]
}

func (c *Conversation) GetMessagesByRole(role Role) []*Message {
	var result []*Message
	for _, m := range c.Messages {
		if m.Role == role {
			result = append(result, m)
		}
	}
	return result
}

func (c *Conversation) Clear() {
	c.Messages = make([]*Message, 0)
	c.UpdatedAt = time.Now()
}
