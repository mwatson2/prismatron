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

      {/* Overview */}
      <section className="content-card">
        <h2 className="subsection-title">Overview</h2>
        <div className="space-y-4 text-metal-silver leading-relaxed">
          <p>
            The system software for Prismatron comes to about 50,000 lines of code, mostly written
            by Claude Code. There's an additional 20,000 lines of tests. The server code is written
            in Python and we use Vite and React for the web controller. The repository is hosted on
            GitHub and this website served by GitHub Pages. We use an AWS CodeBuild server to run
            CI with GPU support.
          </p>
          <p>
            This was mostly created in evenings and weekends over about 4-5 months, with some final
            touches over a further 3 months. A project of this size would take much much longer
            without AI support, to the point of being impractical as a hobby project, at least for me.
          </p>
        </div>
      </section>

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
            <h3 className="font-retro text-neon-purple mb-2">Playlist Controller</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• Manages progression through a playlist of image sources</li>
              <li>• Handles playlist updates</li>
              <li>• Exposes API for mobile app control</li>
            </ul>
          </div>

          <div className="border-l-2 border-neon-blue pl-4">
            <h3 className="font-retro text-neon-blue mb-2">Image sources</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• Source components for images, videos and effects</li>
              <li>• Each source generates a sequence of frames when asked by the Playlist controller</li>
            </ul>
          </div>

          <div className="border-l-2 border-neon-red pl-4">
            <h3 className="font-retro text-neon-red mb-2">Renderer</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• Outputs the optimized frames per frame timestamps</li>
              <li>• Applies visual effects and transformations based on audio analysis</li>
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

          <div className="border-l-2 border-neon-green pl-4">
              <h3 className="font-retro text-neon-green mb-2">API server and control site</h3>
              <ul className="text-metal-silver space-y-1 text-sm">
                <li>• The API server provides full control of the system, playlist configuration, effects configuration, system monitoring etc.</li>
                <li>• Used by the controller web app (see below)</li>
              </ul>
            </div>
          </div>
      </section>

      {/* Mobile Control App */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Smartphone size={20} />
          Control App
        </h2>
        <p className="text-metal-silver mb-6">
          A web-based control app (works on any device with a browser) provides full control through six main screens:
        </p>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-dark-800 p-4 rounded border border-neon-cyan/20">
            <h4 className="font-retro text-neon-cyan text-sm mb-3">Home — Preview & Control</h4>
            <ul className="text-metal-silver text-sm space-y-1">
              <li>• Live LED preview canvas showing current output</li>
              <li>• Playback controls (play/pause/next/previous)</li>
              <li>• Current item and duration display</li>
              <li>• Status indicators (CPU, GPU, FPS, temperatures)</li>
              <li>• Real-time audio visualizer</li>
              <li>• Control for optimization interations</li>
            </ul>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-pink/20">
            <h4 className="font-retro text-neon-pink text-sm mb-3">Upload — Add Content</h4>
            <ul className="text-metal-silver text-sm space-y-1">
              <li>• Drag-and-drop file upload</li>
              <li>• Multi-file batch upload support</li>
              <li>• Automatic video conversion with progress tracking</li>
              <li>• Image and video format support</li>
            </ul>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-green/20">
            <h4 className="font-retro text-neon-green text-sm mb-3">Media — Manage Files</h4>
            <ul className="text-metal-silver text-sm space-y-1">
              <li>• Browse existing media library</li>
              <li>• Manage uploaded files separately</li>
              <li>• Add files to playlist with one tap</li>
              <li>• Rename and delete files</li>
            </ul>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-purple/20">
            <h4 className="font-retro text-neon-purple text-sm mb-3">Effects — Visual Effects</h4>
            <ul className="text-metal-silver text-sm space-y-1">
              <li>• Browse effects by category</li>
              <li>• Configure effect parameters (colors, speed, etc.)</li>
              <li>• Add effects to playlist with custom duration</li>
              <li>• Audio-reactive mode configuration</li>
            </ul>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-orange/20">
            <h4 className="font-retro text-neon-orange text-sm mb-3">Playlist — Manage Queue</h4>
            <ul className="text-metal-silver text-sm space-y-1">
              <li>• Drag-and-drop reordering</li>
              <li>• Save and load named playlists</li>
              <li>• Auto-repeat and shuffle modes</li>
              <li>• Configure transitions between items</li>
            </ul>
          </div>

          <div className="bg-dark-800 p-4 rounded border border-neon-blue/20">
            <h4 className="font-retro text-neon-blue text-sm mb-3">Settings — System Config</h4>
            <ul className="text-metal-silver text-sm space-y-1">
              <li>• Global brightness control</li>
              <li>• WiFi network management</li>
              <li>• Audio source selection (mic/test file)</li>
              <li>• System restart and reboot</li>
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
            Accepts API commands for real-time control
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            Streams status updates and preview images Add back to the app over a WebSocket
          </li>
        </ul>

        <p className="text-neon-green mt-6 font-bold">
          No internet connection required—works entirely on local network.
        </p>
      </section>
    </motion.div>
  )
}
