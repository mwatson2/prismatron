import { motion } from 'framer-motion'
import { Scale, Mail, ExternalLink } from 'lucide-react'

export default function LicensePage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <Scale className="text-neon-cyan" />
        License
      </h1>

      {/* Overview */}
      <section className="content-card">
        <p className="text-metal-silver text-lg mb-4">
          Prismatron is open for non-commercial use. You're free to use, modify,
          and share this project for:
        </p>

        <ul className="space-y-2 text-metal-silver mb-6">
          <li className="flex items-center gap-2">
            <span className="text-neon-green">✓</span>
            Personal projects
          </li>
          <li className="flex items-center gap-2">
            <span className="text-neon-green">✓</span>
            Research and education
          </li>
          <li className="flex items-center gap-2">
            <span className="text-neon-green">✓</span>
            Art installations
          </li>
          <li className="flex items-center gap-2">
            <span className="text-neon-green">✓</span>
            Anything non-commercial
          </li>
        </ul>
      </section>

      {/* License Details */}
      <section className="content-card">
        <h2 className="subsection-title">License Details</h2>

        <div className="space-y-4">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-2">Code</h3>
            <p className="text-metal-silver text-sm mb-2">
              Python, C++, CUDA kernels, and all source code is licensed under the
            </p>
            <a
              href="https://polyformproject.org/licenses/noncommercial/1.0.0/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-neon-cyan hover:text-neon-pink transition-colors"
            >
              Polyform Noncommercial License 1.0.0
              <ExternalLink size={14} />
            </a>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-2">Everything Else</h3>
            <p className="text-metal-silver text-sm mb-2">
              Documentation, schematics, images, and designs are licensed under
            </p>
            <a
              href="https://creativecommons.org/licenses/by-nc/4.0/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-neon-pink hover:text-neon-cyan transition-colors"
            >
              Creative Commons BY-NC 4.0
              <ExternalLink size={14} />
            </a>
          </div>
        </div>
      </section>

      {/* Attribution */}
      <section className="content-card">
        <h2 className="subsection-title">Attribution</h2>
        <p className="text-metal-silver mb-4">
          If you build on this work, please credit:
        </p>

        <div className="bg-dark-800 p-4 rounded border border-neon-cyan/30">
          <p className="text-neon-cyan font-mono text-sm">
            Prismatron by Mark Watson —{' '}
            <a
              href="https://prismatron.info"
              className="text-neon-pink hover:underline"
            >
              prismatron.info
            </a>
          </p>
        </div>
      </section>

      {/* Commercial Use */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Mail size={20} />
          Commercial Use
        </h2>
        <p className="text-metal-silver mb-4">
          Want to use Prismatron in a commercial product or service? Get in touch
          and we can work something out.
        </p>

        <a
          href="mailto:markwatson@cantab.net"
          className="inline-flex items-center gap-2 px-4 py-2 bg-dark-700 hover:bg-dark-600 border border-neon-pink/50 hover:border-neon-pink rounded transition-all text-neon-pink"
        >
          <Mail size={18} />
          markwatson@cantab.net
        </a>
      </section>
    </motion.div>
  )
}
