import { motion } from 'framer-motion'
import { Settings, Smartphone, Layers } from 'lucide-react'

export default function SoftwarePage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <Settings className="text-neon-cyan" />
        Software Architecture
      </h1>

      {/* Core Components */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Layers size={20} />
          Core Components
        </h2>

        <div className="space-y-6">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-2">Calibration Tool</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• Automated LED sequencing and photo capture</li>
              <li>• Pattern extraction and storage</li>
              <li>• Position validation and correction</li>
            </ul>
          </div>

          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-2">Optimization Engine</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• GPU-accelerated matrix operations</li>
              <li>• Multiple solver strategies (quality vs. speed tradeoffs)</li>
              <li>• Frame caching and warm-start support</li>
            </ul>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-2">Audio Analyzer</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• Real-time FFT and spectral analysis via Aubio</li>
              <li>• Beat and tempo tracking</li>
              <li>• Pattern recognition for EDM structures</li>
            </ul>
          </div>

          <div className="border-l-2 border-neon-purple pl-4">
            <h3 className="font-retro text-neon-purple mb-2">Mode Controller</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• Switches between display modes</li>
              <li>• Handles transitions and blending</li>
              <li>• Exposes API for mobile app control</li>
            </ul>
          </div>

          <div className="border-l-2 border-neon-orange pl-4">
            <h3 className="font-retro text-neon-orange mb-2">WLED on QuinLED DigiOcta</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• Receives LED data via UDP/DDP protocol</li>
              <li>• 8 parallel LED data outputs for fast refresh</li>
              <li>• Web-based configuration and diagnostics</li>
              <li>• Fallback effects when Jetson not connected</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Mobile Control App */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Smartphone size={20} />
          Mobile Control App
        </h2>
        <p className="text-metal-silver mb-6">
          A companion mobile app (web-based, works on any device) provides full control:
        </p>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-dark-800 p-4 rounded border border-neon-cyan/20">
            <h4 className="font-retro text-neon-cyan text-sm mb-3">Display Modes</h4>
            <ul className="text-metal-silver text-sm space-y-1">
              <li>• Static image display</li>
              <li>• Audio-reactive mode</li>
              <li>• Demo/showcase cycling</li>
              <li>• Manual color controls</li>
            </ul>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-pink/20">
            <h4 className="font-retro text-neon-pink text-sm mb-3">Audio Settings</h4>
            <ul className="text-metal-silver text-sm space-y-1">
              <li>• Input source selection</li>
              <li>• Sensitivity curves</li>
              <li>• Visual mapping customization</li>
              <li>• Beat detection tuning</li>
            </ul>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-green/20">
            <h4 className="font-retro text-neon-green text-sm mb-3">System Status</h4>
            <ul className="text-metal-silver text-sm space-y-1">
              <li>• Temperature monitoring</li>
              <li>• FPS and performance</li>
              <li>• Connection status</li>
              <li>• Error logging</li>
            </ul>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-purple/20">
            <h4 className="font-retro text-neon-purple text-sm mb-3">Calibration</h4>
            <ul className="text-metal-silver text-sm space-y-1">
              <li>• Guided calibration wizard</li>
              <li>• Pattern preview</li>
              <li>• Manual LED testing</li>
              <li>• Validation tools</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Connectivity */}
      <section className="content-card">
        <h2 className="subsection-title">Connectivity</h2>
        <p className="text-metal-silver mb-4">
          The mobile app connects to Prismatron over WiFi. The Jetson runs a lightweight
          web server that:
        </p>

        <ul className="space-y-2 text-metal-silver">
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            Serves the control interface
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            Accepts WebSocket commands for real-time control
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            Streams status updates back to the app
          </li>
        </ul>

        <p className="text-neon-green mt-6 font-bold">
          No internet connection required—works entirely on local network.
        </p>
      </section>
    </motion.div>
  )
}
