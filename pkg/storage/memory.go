package storage

import (
	"fmt"
	"time"
)

// ConversationMemory represents a stored conversation for training/reference.
type ConversationMemory struct {
	ID        string          `json:"id"`
	Domain    string          `json:"domain"`
	SessionID string          `json:"session_id"`
	Messages  []MemoryMessage `json:"messages"`
	Summary   string          `json:"summary,omitempty"`
	Quality   float64         `json:"quality,omitempty"`
	Outcome   string          `json:"outcome,omitempty"` // success, failure, partial
	Tags      []string        `json:"tags,omitempty"`
	Metadata  map[string]any  `json:"metadata,omitempty"`
	CreatedAt time.Time       `json:"created_at"`
	UpdatedAt time.Time       `json:"updated_at"`
}

// MemoryMessage represents a message in a conversation.
type MemoryMessage struct {
	Role      string    `json:"role"` // user, assistant, system
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
	Tokens    int       `json:"tokens,omitempty"`
}

// MemoryStore manages conversation memory persistence.
type MemoryStore struct {
	storage *Storage
}

// NewMemoryStore creates a new memory store.
func NewMemoryStore(storage *Storage) *MemoryStore {
	return &MemoryStore{storage: storage}
}

// Save saves a conversation memory.
func (s *MemoryStore) Save(memory *ConversationMemory) error {
	if memory.ID == "" {
		memory.ID = generateID()
	}
	if memory.CreatedAt.IsZero() {
		memory.CreatedAt = time.Now()
	}
	memory.UpdatedAt = time.Now()

	path := fmt.Sprintf("memory/%s/%s.json", memory.Domain, memory.ID)
	return s.storage.SaveJSON(path, memory)
}

// Get retrieves a conversation memory by ID.
func (s *MemoryStore) Get(domain, id string) (*ConversationMemory, error) {
	path := fmt.Sprintf("memory/%s/%s.json", domain, id)

	var memory ConversationMemory
	if err := s.storage.LoadJSON(path, &memory); err != nil {
		return nil, err
	}
	return &memory, nil
}

// List lists all conversation memories for a domain.
func (s *MemoryStore) List(domain string) ([]*ConversationMemory, error) {
	dir := fmt.Sprintf("memory/%s", domain)
	files, err := s.storage.List(dir)
	if err != nil {
		return nil, err
	}

	var memories []*ConversationMemory
	for _, file := range files {
		var memory ConversationMemory
		path := fmt.Sprintf("memory/%s/%s", domain, file)
		if err := s.storage.LoadJSON(path, &memory); err != nil {
			continue
		}
		memories = append(memories, &memory)
	}
	return memories, nil
}

// GetRecent gets the most recent conversation memories.
func (s *MemoryStore) GetRecent(domain string, limit int) ([]*ConversationMemory, error) {
	memories, err := s.List(domain)
	if err != nil {
		return nil, err
	}

	// Sort by created_at descending (simple bubble sort for small lists)
	for i := 0; i < len(memories)-1; i++ {
		for j := i + 1; j < len(memories); j++ {
			if memories[j].CreatedAt.After(memories[i].CreatedAt) {
				memories[i], memories[j] = memories[j], memories[i]
			}
		}
	}

	if len(memories) > limit {
		memories = memories[:limit]
	}
	return memories, nil
}

// GetSuccessful gets successful conversations for training.
func (s *MemoryStore) GetSuccessful(domain string, limit int) ([]*ConversationMemory, error) {
	memories, err := s.List(domain)
	if err != nil {
		return nil, err
	}

	var successful []*ConversationMemory
	for _, m := range memories {
		if m.Outcome == "success" {
			successful = append(successful, m)
			if len(successful) >= limit {
				break
			}
		}
	}
	return successful, nil
}

// Delete removes a conversation memory.
func (s *MemoryStore) Delete(domain, id string) error {
	path := fmt.Sprintf("memory/%s/%s.json", domain, id)
	return s.storage.Delete(path)
}

// SearchByTags finds memories with matching tags.
func (s *MemoryStore) SearchByTags(domain string, tags []string) ([]*ConversationMemory, error) {
	memories, err := s.List(domain)
	if err != nil {
		return nil, err
	}

	var matching []*ConversationMemory
	for _, m := range memories {
		for _, tag := range tags {
			for _, mTag := range m.Tags {
				if tag == mTag {
					matching = append(matching, m)
					break
				}
			}
		}
	}
	return matching, nil
}

// ConvertToTrainingExample converts a successful memory to a training example.
func (s *MemoryStore) ConvertToTrainingExample(memory *ConversationMemory) *TrainingExample {
	if len(memory.Messages) < 2 {
		return nil
	}

	// Find the main user input and assistant output
	var input, output string
	for _, msg := range memory.Messages {
		if msg.Role == "user" && input == "" {
			input = msg.Content
		}
		if msg.Role == "assistant" {
			output = msg.Content
		}
	}

	return &TrainingExample{
		ID:        generateID(),
		Domain:    memory.Domain,
		Category:  "conversation",
		Input:     input,
		Output:    output,
		Quality:   memory.Quality,
		Validated: memory.Outcome == "success",
		Tags:      memory.Tags,
		Metadata: map[string]any{
			"source_memory_id": memory.ID,
			"session_id":       memory.SessionID,
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}
