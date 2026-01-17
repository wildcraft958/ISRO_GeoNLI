import { useState } from 'react';
import {
  checkAllEndpoints,
  ConnectivityResult,
} from '../utils/api';

interface DiagnosticResult {
  name: string;
  status: 'pass' | 'fail' | 'warning' | 'pending';
  message: string;
  details?: any;
}

export default function DiagnosticPage() {
  const [results, setResults] = useState<DiagnosticResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const addResult = (result: DiagnosticResult) => {
    setResults((prev) => [...prev, result]);
  };

  const clearResults = () => {
    setResults([]);
  };

  const checkCanvasSupport = (): DiagnosticResult => {
    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        return {
          name: 'Canvas 2D Context',
          status: 'fail',
          message: 'Canvas 2D context is not supported in this browser',
        };
      }

      // Test basic drawing operations
      canvas.width = 100;
      canvas.height = 100;
      ctx.fillStyle = '#FF0000';
      ctx.fillRect(0, 0, 50, 50);

      const imageData = ctx.getImageData(0, 0, 50, 50);
      const hasRed = imageData.data[0] === 255;

      if (!hasRed) {
        return {
          name: 'Canvas 2D Context',
          status: 'warning',
          message: 'Canvas is available but drawing operations may not work correctly',
        };
      }

      return {
        name: 'Canvas 2D Context',
        status: 'pass',
        message: 'Canvas 2D context is fully supported',
        details: {
          width: canvas.width,
          height: canvas.height,
        },
      };
    } catch (error) {
      return {
        name: 'Canvas 2D Context',
        status: 'fail',
        message: `Canvas error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  };

  const checkLocalStorage = (): DiagnosticResult => {
    try {
      if (typeof Storage === 'undefined') {
        return {
          name: 'LocalStorage',
          status: 'fail',
          message: 'LocalStorage is not supported in this browser',
        };
      }

      const testKey = '__sam3_diagnostic_test__';
      localStorage.setItem(testKey, 'test');
      const value = localStorage.getItem(testKey);
      localStorage.removeItem(testKey);

      if (value !== 'test') {
        return {
          name: 'LocalStorage',
          status: 'fail',
          message: 'LocalStorage read/write test failed',
        };
      }

      return {
        name: 'LocalStorage',
        status: 'pass',
        message: 'LocalStorage is working correctly',
      };
    } catch (error) {
      return {
        name: 'LocalStorage',
        status: 'fail',
        message: `LocalStorage error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  };

  const checkBrowserInfo = (): DiagnosticResult => {
    const userAgent = navigator.userAgent;
    const language = navigator.language;
    const platform = navigator.platform;
    const cookieEnabled = navigator.cookieEnabled;
    const onLine = navigator.onLine;

    return {
      name: 'Browser Information',
      status: 'pass',
      message: 'Browser information retrieved',
      details: {
        userAgent,
        language,
        platform,
        cookieEnabled,
        onLine,
      },
    };
  };

  const checkImageLoading = async (): Promise<DiagnosticResult> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        resolve({
          name: 'Image Loading',
          status: 'pass',
          message: 'Images can be loaded successfully',
          details: {
            width: img.width,
            height: img.height,
          },
        });
      };
      img.onerror = () => {
        resolve({
          name: 'Image Loading',
          status: 'fail',
          message: 'Failed to load test image',
        });
      };
      // Use a small data URL as test image
      img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzAwMCIvPjwvc3ZnPg==';
    });
  };

  const formatConnectivityResult = (result: ConnectivityResult, name: string): DiagnosticResult => {
    if (result.reachable) {
      return {
        name,
        status: result.statusCode && result.statusCode >= 400 ? 'warning' : 'pass',
        message: result.statusCode
          ? `Endpoint is reachable (HTTP ${result.statusCode})`
          : 'Endpoint is reachable',
        details: {
          endpoint: result.endpoint,
          statusCode: result.statusCode,
          statusText: result.statusText,
          responseTime: result.responseTime ? `${result.responseTime}ms` : undefined,
        },
      };
    } else {
      return {
        name,
        status: 'fail',
        message: `Endpoint is not reachable: ${result.error || 'Unknown error'}`,
        details: {
          endpoint: result.endpoint,
          error: result.error,
          errorType: result.errorType,
          responseTime: result.responseTime ? `${result.responseTime}ms` : undefined,
        },
      };
    }
  };

  const runAllDiagnostics = async () => {
    setIsRunning(true);
    clearResults();

    // 1. Browser Information
    addResult(checkBrowserInfo());

    // 2. Canvas Support
    addResult(checkCanvasSupport());

    // 3. LocalStorage
    addResult(checkLocalStorage());

    // 4. Image Loading
    const imageResult = await checkImageLoading();
    addResult(imageResult);

    // 5. API Endpoint Connectivity
    addResult({
      name: 'API Endpoint Connectivity',
      status: 'pending',
      message: 'Testing connectivity...',
    });

    try {
      const connectivityResults = await checkAllEndpoints(5000);
      setResults((prev) => {
        const newResults = [...prev];
        const index = newResults.findIndex((r) => r.name === 'API Endpoint Connectivity');
        if (index !== -1) {
          newResults[index] = formatConnectivityResult(
            connectivityResults.apiEndpoint,
            'API Endpoint Connectivity'
          );
        }
        return newResults;
      });

      addResult(
        formatConnectivityResult(connectivityResults.inferEndpoint, 'Infer Endpoint Connectivity')
      );
    } catch (error) {
      setResults((prev) => {
        const newResults = [...prev];
        const index = newResults.findIndex((r) => r.name === 'API Endpoint Connectivity');
        if (index !== -1) {
          newResults[index] = {
            name: 'API Endpoint Connectivity',
            status: 'fail',
            message: `Error testing connectivity: ${error instanceof Error ? error.message : 'Unknown error'}`,
          };
        }
        return newResults;
      });
    }

    setIsRunning(false);
  };

  const getStatusIcon = (status: DiagnosticResult['status']) => {
    switch (status) {
      case 'pass':
        return '✅';
      case 'fail':
        return '❌';
      case 'warning':
        return '⚠️';
      case 'pending':
        return '⏳';
      default:
        return '❓';
    }
  };

  const getStatusColor = (status: DiagnosticResult['status']) => {
    switch (status) {
      case 'pass':
        return '#10b981';
      case 'fail':
        return '#ef4444';
      case 'warning':
        return '#f59e0b';
      case 'pending':
        return '#64748b';
      default:
        return '#64748b';
    }
  };

  return (
    <div className="diagnostic-page">
      <div className="panel-section">
        <h2>Diagnostic Tests</h2>
        <p className="hint">
          Run diagnostic tests to check browser compatibility, API connectivity, and system configuration.
        </p>
        <div className="button-group" style={{ marginTop: '1rem' }}>
          <button
            className="segment-button"
            onClick={runAllDiagnostics}
            disabled={isRunning}
          >
            {isRunning ? 'Running Tests...' : 'Run All Diagnostics'}
          </button>
          <button
            className="stop-button"
            onClick={clearResults}
            disabled={isRunning}
          >
            Clear Results
          </button>
        </div>
      </div>

      {results.length > 0 && (
        <div className="panel-section">
          <h2>Diagnostic Results</h2>
          <div className="diagnostic-results">
            {results.map((result, index) => (
              <div
                key={index}
                className="diagnostic-result"
                style={{
                  borderLeft: `4px solid ${getStatusColor(result.status)}`,
                }}
              >
                <div className="result-header">
                  <span className="result-icon">{getStatusIcon(result.status)}</span>
                  <strong>{result.name}</strong>
                  <span
                    className="result-status"
                    style={{ color: getStatusColor(result.status) }}
                  >
                    {result.status.toUpperCase()}
                  </span>
                </div>
                <div className="result-message">{result.message}</div>
                {result.details && (
                  <details className="result-details">
                    <summary>Details</summary>
                    <pre>{JSON.stringify(result.details, null, 2)}</pre>
                  </details>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

