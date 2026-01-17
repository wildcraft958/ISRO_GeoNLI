import { useState, useEffect } from 'react';
import { LLMConfig } from '../utils/api';

interface LLMConfigFormProps {
  onConfigChange: (config: LLMConfig) => void;
  initialConfig?: Partial<LLMConfig>;
}

// Preset configurations for common providers
const PRESETS = {
  openai: {
    base_url: 'https://api.openai.com/v1',
    model: 'gpt-4o',
    name: 'openai-gpt4o',
  },
  vllm_modal: {
    base_url: 'https://srinjoy59--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1',
    model: 'Qwen/Qwen3-VL-30B-A3B-Instruct',
    name: 'qwen3-vl-30b-modal',
  },
};

export default function LLMConfigForm({ onConfigChange, initialConfig }: LLMConfigFormProps) {
  const [config, setConfig] = useState<LLMConfig>({
    base_url: initialConfig?.base_url || PRESETS.vllm_modal.base_url,
    model: initialConfig?.model || PRESETS.vllm_modal.model,
    api_key: initialConfig?.api_key || '',
    name: initialConfig?.name || PRESETS.vllm_modal.name,
    max_tokens: initialConfig?.max_tokens || 2048,
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string>('vllm_modal');

  // Notify parent of initial config on mount
  useEffect(() => {
    onConfigChange(config);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handlePresetChange = (preset: string) => {
    setSelectedPreset(preset);
    if (preset === 'openai') {
      const newConfig = {
        ...config,
        ...PRESETS.openai,
        api_key: config.api_key, // Preserve API key
      };
      setConfig(newConfig);
      onConfigChange(newConfig);
    } else if (preset === 'vllm_modal') {
      const newConfig = {
        ...config,
        ...PRESETS.vllm_modal,
        api_key: '', // vLLM doesn't need API key
      };
      setConfig(newConfig);
      onConfigChange(newConfig);
    }
  };

  const handleChange = (field: keyof LLMConfig, value: string | number) => {
    const newConfig = { ...config, [field]: value };
    setConfig(newConfig);
    onConfigChange(newConfig);
  };

  return (
    <div className="llm-config-form">
      <h3>LLM Configuration</h3>
      
      <div className="form-group">
        <label htmlFor="preset">Provider Preset</label>
        <select
          id="preset"
          value={selectedPreset}
          onChange={(e) => handlePresetChange(e.target.value)}
        >
          <option value="vllm_modal">Qwen3-VL-30B (Modal vLLM)</option>
          <option value="openai">OpenAI GPT-4o</option>
          <option value="custom">Custom</option>
        </select>
      </div>
      
      <div className="form-group">
        <label htmlFor="base_url">Base URL</label>
        <input
          id="base_url"
          type="text"
          value={config.base_url}
          onChange={(e) => {
            handleChange('base_url', e.target.value);
            setSelectedPreset('custom');
          }}
          placeholder="https://api.openai.com/v1"
        />
      </div>

      <div className="form-group">
        <label htmlFor="model">Model</label>
        <input
          id="model"
          type="text"
          value={config.model}
          onChange={(e) => {
            handleChange('model', e.target.value);
            setSelectedPreset('custom');
          }}
          placeholder="gpt-4o"
        />
      </div>

      <div className="form-group">
        <label htmlFor="api_key">API Key {selectedPreset === 'vllm_modal' && <span className="hint">(not required for Modal vLLM)</span>}</label>
        <input
          id="api_key"
          type="password"
          value={config.api_key}
          onChange={(e) => handleChange('api_key', e.target.value)}
          placeholder={selectedPreset === 'vllm_modal' ? '(optional)' : 'sk-...'}
        />
      </div>

      <button
        type="button"
        className="toggle-advanced"
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        {showAdvanced ? 'Hide' : 'Show'} Advanced Options
      </button>

      {showAdvanced && (
        <>
          <div className="form-group">
            <label htmlFor="name">Name (Optional)</label>
            <input
              id="name"
              type="text"
              value={config.name || ''}
              onChange={(e) => handleChange('name', e.target.value)}
              placeholder="openai-gpt4o"
            />
          </div>

          <div className="form-group">
            <label htmlFor="max_tokens">Max Tokens</label>
            <input
              id="max_tokens"
              type="number"
              value={config.max_tokens ?? 2048}
              onChange={(e) => {
                const value = e.target.value;
                if (value === '') {
                  // Allow empty input temporarily
                  handleChange('max_tokens', undefined as any);
                } else {
                  const numValue = parseInt(value, 10);
                  if (!isNaN(numValue) && numValue > 0) {
                    handleChange('max_tokens', numValue);
                  }
                }
              }}
              onBlur={(e) => {
                // On blur, if empty or invalid, set to default
                const value = e.target.value;
                if (value === '' || isNaN(parseInt(value, 10)) || parseInt(value, 10) <= 0) {
                  handleChange('max_tokens', 2048);
                }
              }}
              min="1"
              max="32768"
            />
          </div>
        </>
      )}
    </div>
  );
}

