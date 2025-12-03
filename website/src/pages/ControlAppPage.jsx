import { motion } from 'framer-motion'
import { Smartphone, Sliders, BarChart3, Image, Music, Wifi } from 'lucide-react'

export default function ControlAppPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <Smartphone className="text-neon-pink" />
        Control App
      </h1>

      {/* Overview */}
      <section className="content-card">
        <p className="text-metal-silver mb-4">
          Prismatron includes a web-based control interface that runs directly on the Jetson.
          Access it from any device on the same network—phone, tablet, or laptop. The retro-futurism
          design matches the display's aesthetic, with neon accents and a dark theme optimized
          for use in low-light environments.
        </p>
        <p className="text-metal-silver">
          The interface provides real-time control over all display modes, audio settings,
          and system monitoring—no app installation required.
        </p>

        <div className="mt-6 p-4 retro-panel rounded border border-neon-pink/30">
          <div className="text-neon-pink/50 text-sm font-mono mb-2">[SCREENSHOT PLACEHOLDER]</div>
          <p className="text-metal-silver text-sm">Control app dashboard on mobile device</p>
        </div>
      </section>

      {/* Display Modes */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Image size={20} />
          Display Modes
        </h2>
        <p className="text-metal-silver mb-4">
          Switch between different display modes with a single tap:
        </p>

        <div className="space-y-4">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-1">Image Display</h3>
            <p className="text-metal-silver text-sm">
              Upload images or select from a library. The optimization engine renders them
              in real-time, with controls for brightness, contrast, and color adjustment.
            </p>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-1">Audio-Reactive</h3>
            <p className="text-metal-silver text-sm">
              Responds to music in real-time. Adjust sensitivity, choose color palettes,
              and tune how build-ups and drops affect the display.
            </p>
          </div>

          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-1">Video Playback</h3>
            <p className="text-metal-silver text-sm">
              Play video content optimized for the display. Supports playlist mode
              with crossfade transitions between clips.
            </p>
          </div>

          <div className="border-l-2 border-neon-purple pl-4">
            <h3 className="font-retro text-neon-purple mb-1">Demo Mode</h3>
            <p className="text-metal-silver text-sm">
              Cycles through preset patterns and effects automatically. Perfect for
              showcasing the display without manual intervention.
            </p>
          </div>
        </div>
      </section>

      {/* Audio Controls */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Music size={20} />
          Audio Controls
        </h2>
        <p className="text-metal-silver mb-4">
          Fine-tune how the display responds to audio:
        </p>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-dark-800 p-4 rounded border border-neon-cyan/20">
            <h4 className="font-retro text-neon-cyan text-sm mb-2">Input Selection</h4>
            <p className="text-metal-silver text-sm">
              Choose between line-in, microphone, or Bluetooth audio sources.
            </p>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-pink/20">
            <h4 className="font-retro text-neon-pink text-sm mb-2">Sensitivity</h4>
            <p className="text-metal-silver text-sm">
              Adjust how strongly the display reacts to different volume levels.
            </p>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-green/20">
            <h4 className="font-retro text-neon-green text-sm mb-2">Color Palette</h4>
            <p className="text-metal-silver text-sm">
              Select from preset palettes or create custom color schemes.
            </p>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-purple/20">
            <h4 className="font-retro text-neon-purple text-sm mb-2">Effect Intensity</h4>
            <p className="text-metal-silver text-sm">
              Control how dramatic the build-up and drop effects appear.
            </p>
          </div>
        </div>
      </section>

      {/* System Monitoring */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <BarChart3 size={20} />
          System Monitoring
        </h2>
        <p className="text-metal-silver mb-4">
          Real-time status and performance metrics:
        </p>

        <div className="bg-dark-800 rounded border border-neon-cyan/20 overflow-hidden">
          <table className="spec-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="text-neon-cyan">FPS</td>
                <td>Current frame rate of the optimization engine</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">CPU/GPU Temp</td>
                <td>Jetson thermal monitoring with alerts</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">Audio Level</td>
                <td>Real-time input level meter with AGC status</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">BPM</td>
                <td>Detected beats per minute from audio</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">LED Controller</td>
                <td>Connection status to WLED/DigiOcta</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">Network</td>
                <td>WiFi signal strength and AP fallback status</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Settings */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Sliders size={20} />
          Settings
        </h2>

        <div className="space-y-4">
          <div className="border-l-2 border-neon-orange pl-4">
            <h3 className="font-retro text-neon-orange mb-1">Network Configuration</h3>
            <p className="text-metal-silver text-sm">
              Configure WiFi networks. The system automatically falls back to AP mode
              if no known network is available, ensuring you can always connect.
            </p>
          </div>

          <div className="border-l-2 border-neon-yellow pl-4">
            <h3 className="font-retro text-neon-yellow mb-1">Calibration Tools</h3>
            <p className="text-metal-silver text-sm">
              Run the LED calibration wizard to capture diffusion patterns. Includes
              manual LED testing and pattern validation tools.
            </p>
          </div>

          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-1">System Controls</h3>
            <p className="text-metal-silver text-sm">
              Reboot, shutdown, or update the system. View logs and diagnostic information.
            </p>
          </div>
        </div>
      </section>

      {/* Connectivity */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Wifi size={20} />
          Connectivity
        </h2>
        <p className="text-metal-silver mb-4">
          The control interface works entirely over local WiFi:
        </p>

        <ul className="space-y-2 text-metal-silver">
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            <span>Connect to the same network as Prismatron</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            <span>Navigate to <code className="text-neon-pink">http://prismatron.local</code> in any browser</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            <span>Real-time updates via WebSocket—no page refresh needed</span>
          </li>
        </ul>

        <p className="text-neon-green mt-6 font-bold">
          No internet connection required—works entirely on local network.
        </p>

        <div className="mt-4 p-4 bg-dark-800 rounded border border-neon-orange/30">
          <h4 className="font-retro text-neon-orange text-sm mb-2">AP Fallback Mode</h4>
          <p className="text-metal-silver text-sm">
            If no known WiFi network is available, Prismatron creates its own access point
            (<code className="text-neon-cyan">Prismatron-AP</code>). Connect directly to configure
            network settings or control the display.
          </p>
        </div>
      </section>
    </motion.div>
  )
}
