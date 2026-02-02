package prompt

import (
	"bytes"
	"text/template"
)

// Template is a wrapper around Go's text/template for easy prompting.
type Template struct {
	tmpl *template.Template
}

// NewTemplate creates a new prompt template from a string.
func NewTemplate(name, text string) (*Template, error) {
	t, err := template.New(name).Parse(text)
	if err != nil {
		return nil, err
	}
	return &Template{tmpl: t}, nil
}

// Execute renders the template with the given data.
func (t *Template) Execute(data interface{}) (string, error) {
	var buf bytes.Buffer
	if err := t.tmpl.Execute(&buf, data); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// MustNewTemplate helper that panics on error (useful for global constants).
func MustNewTemplate(name, text string) *Template {
	t, err := NewTemplate(name, text)
	if err != nil {
		panic(err)
	}
	return t
}
