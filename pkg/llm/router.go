package llm

// ModelTier represents the capability level of a model.
type ModelTier string

const (
	TierFast      ModelTier = "fast"      // e.g. gpt-4o-mini
	TierSmart     ModelTier = "smart"     // e.g. gpt-4o
	TierReasoning ModelTier = "reasoning" // e.g. o1, deepseek-r1
	TierCoding    ModelTier = "coding"    // e.g. claude-3-5-sonnet
)

// RouterConfig holds providers for different tiers.
type RouterConfig struct {
	Fast      Provider
	Smart     Provider
	Reasoning Provider
	Coding    Provider
}

// Router manages multiple LLM providers and routes based on tier.
type Router struct {
	Config RouterConfig
}

// NewRouter creates a router with a detailed config.
func NewRouter(config RouterConfig) *Router {
	// Fallbacks
	if config.Smart == nil {
		config.Smart = config.Fast
	}
	if config.Fast == nil {
		config.Fast = config.Smart
	}
	if config.Reasoning == nil {
		config.Reasoning = config.Smart
	}
	if config.Coding == nil {
		config.Coding = config.Smart
	}

	// If everything is nil, this will panic later, but we assume at least one is provided
	// and propagated via fallbacks logic (simplified here).

	return &Router{Config: config}
}

// GetProvider returns the provider for the requested tier.
func (r *Router) GetProvider(tier ModelTier) Provider {
	switch tier {
	case TierFast:
		return r.Config.Fast
	case TierSmart:
		return r.Config.Smart
	case TierReasoning:
		return r.Config.Reasoning
	case TierCoding:
		return r.Config.Coding
	default:
		return r.Config.Smart
	}
}

// RouteByComplexity maps a complexity or intent to a provider.
func (r *Router) RouteByIntent(intent string) Provider {
	switch intent {
	case "coding":
		return r.Config.Coding
	case "reasoning", "complex_logic":
		return r.Config.Reasoning
	case "low", "simple":
		return r.Config.Fast
	default: // "general", "high"
		return r.Config.Smart
	}
}
