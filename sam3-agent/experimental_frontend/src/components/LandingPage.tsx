import { useNavigate } from 'react-router-dom';

export default function LandingPage() {
    const navigate = useNavigate();

    const features = [
        {
            icon: '🎯',
            title: 'Pyramidal Batch Processing',
            description: 'Process 16 tiles simultaneously with 99% reduction in text encoding time. Handles large satellite images efficiently.',
            stats: '10-20× faster'
        },
        {
            icon: '🤖',
            title: 'VLM-Enhanced Pipeline',
            description: 'Three-stage AI verification: prompt refinement, detection verification, and automatic retry with rephrasing.',
            stats: '40-50% fewer false positives'
        },
        {
            icon: '🔄',
            title: 'Provider Agnostic',
            description: 'Works with any OpenAI-compatible API. Switch between GPT-4o, Claude, vLLM, or custom models without code changes.',
            stats: 'Zero vendor lock-in'
        },
        {
            icon: '📊',
            title: 'Multi-Task Capabilities',
            description: 'Single deployment handles counting, area calculation, and full segmentation with oriented bounding boxes.',
            stats: '3 endpoints, 1 service'
        },
        {
            icon: '⚡',
            title: 'Optimized Performance',
            description: 'Text encoding cache, GPU batch processing, and warm containers for sub-second response times.',
            stats: '80-95% GPU utilization'
        },
        {
            icon: '🎨',
            title: 'Advanced UI',
            description: 'Modern React interface with Zustand state management, real-time progress indicators, and advanced configuration.',
            stats: 'Full feature parity'
        }
    ];

    const useCases = [
        {
            title: 'Satellite Image Analysis',
            description: 'Count solar panels, detect buildings, measure coverage across large aerial imagery',
            icon: '🛰️'
        },
        {
            title: 'Industrial Inspection',
            description: 'Segment defects, identify anomalies, generate oriented bounding boxes for rotated objects',
            icon: '🏭'
        },
        {
            title: 'Environmental Monitoring',
            description: 'Track deforestation, count wildlife, measure agricultural areas with GSD support',
            icon: '🌲'
        }
    ];

    return (
        <div className="landing-page">
            {/* Hero Section */}
            <section className="hero-section">
                <div className="hero-content">
                    <div className="hero-badge">
                        <span className="badge-icon">✨</span>
                        <span>Powered by SAM3 + VLM</span>
                    </div>
                    <h1 className="hero-title">
                        Advanced Image Segmentation
                        <span className="gradient-text"> with AI Verification</span>
                    </h1>
                    <p className="hero-description">
                        Production-ready segmentation system with pyramidal batch processing,
                        VLM-enhanced accuracy, and provider-agnostic architecture.
                        Process large satellite images 10-20× faster with 40-50% fewer false positives.
                    </p>
                    <div className="hero-actions">
                        <button
                            className="cta-button primary"
                            onClick={() => navigate('/app')}
                        >
                            Launch Application
                            <span className="button-icon">→</span>
                        </button>
                        <button
                            className="cta-button secondary"
                            onClick={() => window.open('https://github.com/wildcraft958/sam3-agent', '_blank')}
                        >
                            <span className="button-icon">⭐</span>
                            View on GitHub
                        </button>
                    </div>
                    <div className="hero-stats">
                        <div className="stat-item">
                            <div className="stat-value">10-20×</div>
                            <div className="stat-label">Faster Processing</div>
                        </div>
                        <div className="stat-divider"></div>
                        <div className="stat-item">
                            <div className="stat-value">99%</div>
                            <div className="stat-label">Cache Efficiency</div>
                        </div>
                        <div className="stat-divider"></div>
                        <div className="stat-item">
                            <div className="stat-value">40-50%</div>
                            <div className="stat-label">Error Reduction</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Grid */}
            <section className="features-section">
                <div className="section-header-landing">
                    <h2>Key Innovations</h2>
                    <p>Built for production with cutting-edge optimizations</p>
                </div>
                <div className="features-grid">
                    {features.map((feature, idx) => (
                        <div key={idx} className="feature-card">
                            <div className="feature-icon">{feature.icon}</div>
                            <h3>{feature.title}</h3>
                            <p>{feature.description}</p>
                            <div className="feature-stat">{feature.stats}</div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Use Cases */}
            <section className="use-cases-section">
                <div className="section-header-landing">
                    <h2>Use Cases</h2>
                    <p>Proven in real-world applications</p>
                </div>
                <div className="use-cases-grid">
                    {useCases.map((useCase, idx) => (
                        <div key={idx} className="use-case-card">
                            <div className="use-case-icon">{useCase.icon}</div>
                            <h3>{useCase.title}</h3>
                            <p>{useCase.description}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Technical Highlights */}
            <section className="tech-section">
                <div className="tech-content">
                    <h2>Technical Excellence</h2>
                    <div className="tech-grid">
                        <div className="tech-item">
                            <div className="tech-number">01</div>
                            <h4>Text Encoding Cache</h4>
                            <p>Encode once, reuse for all tiles. 100× speedup on text processing.</p>
                        </div>
                        <div className="tech-item">
                            <div className="tech-number">02</div>
                            <h4>Batch Image Encoding</h4>
                            <p>Process 16 tiles simultaneously on GPU. 80-95% utilization.</p>
                        </div>
                        <div className="tech-item">
                            <div className="tech-number">03</div>
                            <h4>VLM Verification</h4>
                            <p>Three-stage pipeline: refine, verify, retry. Minimal token prompts.</p>
                        </div>
                        <div className="tech-item">
                            <div className="tech-number">04</div>
                            <h4>Mask-Based NMS</h4>
                            <p>20-30% better duplicate removal for irregular shapes.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="cta-section">
                <div className="cta-content">
                    <h2>Ready to Get Started?</h2>
                    <p>Deploy locally with Docker or on Modal in minutes</p>
                    <button
                        className="cta-button primary large"
                        onClick={() => navigate('/app')}
                    >
                        Try SAM3 Agent Now
                        <span className="button-icon">→</span>
                    </button>
                </div>
            </section>

            <style>{`
        .landing-page {
          min-height: 100vh;
          background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
          color: #e2e8f0;
        }

        /* Hero Section */
        .hero-section {
          padding: 6rem 2rem;
          max-width: 1200px;
          margin: 0 auto;
          text-align: center;
        }

        .hero-badge {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          background: rgba(59, 130, 246, 0.1);
          border: 1px solid rgba(59, 130, 246, 0.3);
          padding: 0.5rem 1rem;
          border-radius: 2rem;
          font-size: 0.875rem;
          margin-bottom: 2rem;
          backdrop-filter: blur(10px);
        }

        .badge-icon {
          font-size: 1.25rem;
        }

        .hero-title {
          font-size: 3.5rem;
          font-weight: 800;
          line-height: 1.2;
          margin-bottom: 1.5rem;
          background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .gradient-text {
          background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .hero-description {
          font-size: 1.25rem;
          color: #94a3b8;
          max-width: 800px;
          margin: 0 auto 3rem;
          line-height: 1.8;
        }

        .hero-actions {
          display: flex;
          gap: 1rem;
          justify-content: center;
          margin-bottom: 4rem;
        }

        .cta-button {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          padding: 1rem 2rem;
          border-radius: 0.5rem;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s;
          border: none;
        }

        .cta-button.primary {
          background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
          color: white;
          box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
        }

        .cta-button.primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 15px 40px rgba(59, 130, 246, 0.4);
        }

        .cta-button.secondary {
          background: rgba(255, 255, 255, 0.05);
          color: #e2e8f0;
          border: 1px solid rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(10px);
        }

        .cta-button.secondary:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.2);
        }

        .cta-button.large {
          padding: 1.25rem 3rem;
          font-size: 1.125rem;
        }

        .button-icon {
          font-size: 1.25rem;
        }

        .hero-stats {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 2rem;
          padding: 2rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 1rem;
          backdrop-filter: blur(10px);
          max-width: 700px;
          margin: 0 auto;
        }

        .stat-item {
          text-align: center;
        }

        .stat-value {
          font-size: 2rem;
          font-weight: 800;
          background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .stat-label {
          font-size: 0.875rem;
          color: #94a3b8;
          margin-top: 0.25rem;
        }

        .stat-divider {
          width: 1px;
          height: 40px;
          background: rgba(255, 255, 255, 0.1);
        }

        /* Features Section */
        .features-section {
          padding: 4rem 2rem;
          max-width: 1200px;
          margin: 0 auto;
        }

        .section-header-landing {
          text-align: center;
          margin-bottom: 3rem;
        }

        .section-header-landing h2 {
          font-size: 2.5rem;
          font-weight: 700;
          margin-bottom: 0.75rem;
        }

        .section-header-landing p {
          font-size: 1.125rem;
          color: #94a3b8;
        }

        .features-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
          gap: 1.5rem;
        }

        .feature-card {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 1rem;
          padding: 2rem;
          transition: all 0.3s;
          backdrop-filter: blur(10px);
        }

        .feature-card:hover {
          transform: translateY(-5px);
          border-color: rgba(96, 165, 250, 0.5);
          box-shadow: 0 20px 40px rgba(59, 130, 246, 0.2);
        }

        .feature-icon {
          font-size: 3rem;
          margin-bottom: 1rem;
        }

        .feature-card h3 {
          font-size: 1.5rem;
          font-weight: 600;
          margin-bottom: 0.75rem;
        }

        .feature-card p {
          color: #94a3b8;
          line-height: 1.6;
          margin-bottom: 1rem;
        }

        .feature-stat {
          display: inline-block;
          background: rgba(59, 130, 246, 0.1);
          border: 1px solid rgba(59, 130, 246, 0.3);
          padding: 0.5rem 1rem;
          border-radius: 0.5rem;
          font-size: 0.875rem;
          font-weight: 600;
          color: #60a5fa;
        }

        /* Use Cases Section */
        .use-cases-section {
          padding: 4rem 2rem;
          max-width: 1200px;
          margin: 0 auto;
          background: rgba(255, 255, 255, 0.02);
          border-radius: 2rem;
        }

        .use-cases-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 2rem;
        }

        .use-case-card {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 1rem;
          padding: 2rem;
          text-align: center;
          transition: all 0.3s;
        }

        .use-case-card:hover {
          transform: scale(1.05);
          border-color: rgba(167, 139, 250, 0.5);
        }

        .use-case-icon {
          font-size: 4rem;
          margin-bottom: 1rem;
        }

        .use-case-card h3 {
          font-size: 1.25rem;
          font-weight: 600;
          margin-bottom: 0.75rem;
        }

        .use-case-card p {
          color: #94a3b8;
          line-height: 1.6;
        }

        /* Tech Section */
        .tech-section {
          padding: 4rem 2rem;
          max-width: 1200px;
          margin: 0 auto;
        }

        .tech-content h2 {
          text-align: center;
          font-size: 2.5rem;
          font-weight: 700;
          margin-bottom: 3rem;
        }

        .tech-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 2rem;
        }

        .tech-item {
          position: relative;
          padding: 1.5rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 1rem;
          transition: all 0.3s;
        }

        .tech-item:hover {
          border-color: rgba(96, 165, 250, 0.5);
        }

        .tech-number {
          font-size: 3rem;
          font-weight: 800;
          background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          opacity: 0.3;
          position: absolute;
          top: 1rem;
          right: 1rem;
        }

        .tech-item h4 {
          font-size: 1.125rem;
          font-weight: 600;
          margin-bottom: 0.5rem;
        }

        .tech-item p {
          color: #94a3b8;
          font-size: 0.875rem;
          line-height: 1.6;
        }

        /* CTA Section */
        .cta-section {
          padding: 6rem 2rem;
          text-align: center;
        }

        .cta-content {
          max-width: 600px;
          margin: 0 auto;
          padding: 3rem;
          background: rgba(59, 130, 246, 0.1);
          border: 1px solid rgba(59, 130, 246, 0.3);
          border-radius: 2rem;
          backdrop-filter: blur(10px);
        }

        .cta-content h2 {
          font-size: 2.5rem;
          font-weight: 700;
          margin-bottom: 1rem;
        }

        .cta-content p {
          font-size: 1.125rem;
          color: #94a3b8;
          margin-bottom: 2rem;
        }

        @media (max-width: 768px) {
          .hero-title {
            font-size: 2.5rem;
          }

          .hero-description {
            font-size: 1rem;
          }

          .hero-actions {
            flex-direction: column;
          }

          .hero-stats {
            flex-direction: column;
            gap: 1rem;
          }

          .stat-divider {
            width: 100%;
            height: 1px;
          }

          .features-grid,
          .use-cases-grid,
          .tech-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
        </div>
    );
}
