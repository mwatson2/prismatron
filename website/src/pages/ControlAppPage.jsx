import { motion } from 'framer-motion'
import { Smartphone, Home, Upload, Folder, Sparkles, ListMusic, Settings, Wifi } from 'lucide-react'

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
          The interface is a Progressive Web App (PWA) built with React and Vite. It provides
          real-time control through six main screens, with live updates streamed via WebSocket.
          No app installation required—just open a browser.
        </p>

        <div className="mt-6 p-4 retro-panel rounded border border-neon-pink/30">
          <div className="text-neon-pink/50 text-sm font-mono mb-2">[SCREENSHOT PLACEHOLDER]</div>
          <p className="text-metal-silver text-sm">Control app dashboard on mobile device</p>
        </div>
      </section>

      {/* Home - Preview & Control */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Home size={20} />
          Home — Preview & Control
        </h2>
        <p className="text-metal-silver mb-4">
          The home screen is the main dashboard for monitoring and controlling the display in real-time.
        </p>

        <div className="space-y-4">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-1">Live LED Preview</h3>
            <p className="text-metal-silver text-sm">
              A canvas renders the current LED output in real-time, showing exactly what's being
              displayed. The preview uses the actual LED positions from calibration, so you see
              the true scattered layout rather than a grid approximation.
            </p>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-1">Playback Controls</h3>
            <p className="text-metal-silver text-sm">
              Play/pause the current playlist, skip to next or previous items. The current item
              name, type (image/video/effect), and remaining duration are displayed prominently.
            </p>
          </div>

          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-1">System Status</h3>
            <p className="text-metal-silver text-sm">
              Real-time indicators show CPU and GPU utilization, temperatures, current FPS,
              and connection status. Warnings appear if temperatures get too high.
            </p>
          </div>

          <div className="border-l-2 border-neon-purple pl-4">
            <h3 className="font-retro text-neon-purple mb-1">Audio Visualizer</h3>
            <p className="text-metal-silver text-sm">
              When audio-reactive mode is active, a real-time frequency spectrum visualizer
              shows what the system is hearing. Useful for confirming audio input is working.
            </p>
          </div>

          <div className="border-l-2 border-neon-orange pl-4">
            <h3 className="font-retro text-neon-orange mb-1">Optimization Control</h3>
            <p className="text-metal-silver text-sm">
              Adjust the number of optimization iterations in real-time. More iterations mean
              better image quality but lower frame rate—find the right balance for your content.
            </p>
          </div>
        </div>
      </section>

      {/* Upload - Add Content */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Upload size={20} />
          Upload — Add Content
        </h2>
        <p className="text-metal-silver mb-4">
          Add new images and videos to the system directly from your device.
        </p>

        <div className="space-y-4">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-1">Drag and Drop</h3>
            <p className="text-metal-silver text-sm">
              Drop files directly onto the upload zone, or tap to browse. Supports common
              image formats (JPG, PNG, GIF, WebP) and video formats (MP4, MOV, MKV, WebM).
            </p>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-1">Batch Upload</h3>
            <p className="text-metal-silver text-sm">
              Select multiple files at once. Each file uploads sequentially with progress
              indication, and gets added to the uploads folder for later use.
            </p>
          </div>

          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-1">Video Conversion</h3>
            <p className="text-metal-silver text-sm">
              Videos are automatically converted to an optimized format for the display.
              A progress bar shows conversion status, and you're notified when complete.
              The converted video is then available in your media library.
            </p>
          </div>
        </div>
      </section>

      {/* Media - Manage Files */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Folder size={20} />
          Media — Manage Files
        </h2>
        <p className="text-metal-silver mb-4">
          Browse and manage all your media files in one place.
        </p>

        <div className="space-y-4">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-1">Media Library</h3>
            <p className="text-metal-silver text-sm">
              View all images and videos in the permanent media folder. Files are shown
              with thumbnails, names, and file sizes. Tap any file to add it to the current playlist.
            </p>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-1">Uploads Management</h3>
            <p className="text-metal-silver text-sm">
              Recently uploaded files appear in a separate section. Move files you want to
              keep to the permanent media library, or delete temporary uploads to free space.
            </p>
          </div>

          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-1">File Operations</h3>
            <p className="text-metal-silver text-sm">
              Rename files to keep your library organized, or delete files you no longer need.
              When you delete a file, it's automatically removed from any playlists that reference it.
            </p>
          </div>
        </div>
      </section>

      {/* Effects - Visual Effects */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Sparkles size={20} />
          Effects — Visual Effects
        </h2>
        <p className="text-metal-silver mb-4">
          Browse and configure procedural visual effects to add to your playlist.
        </p>

        <div className="space-y-4">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-1">Effect Categories</h3>
            <p className="text-metal-silver text-sm">
              Effects are organized by category: patterns (plasma, fire, matrix), geometric
              (spirals, waves, kaleidoscope), text displays, and audio-reactive visualizations.
              Filter by category to find what you're looking for.
            </p>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-1">Effect Configuration</h3>
            <p className="text-metal-silver text-sm">
              Each effect has configurable parameters—colors, speed, intensity, and more.
              Adjust settings before adding to see how they'll look. Text effects let you
              enter custom messages and choose fonts.
            </p>
          </div>

          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-1">Duration Control</h3>
            <p className="text-metal-silver text-sm">
              Set how long each effect runs before the playlist advances. Effects can run
              for a few seconds or loop indefinitely until you skip manually.
            </p>
          </div>

          <div className="border-l-2 border-neon-purple pl-4">
            <h3 className="font-retro text-neon-purple mb-1">Audio-Reactive Mode</h3>
            <p className="text-metal-silver text-sm">
              Enable audio reactivity for supported effects. Configure sensitivity, frequency
              bands, and how the audio analysis maps to visual parameters like color and intensity.
            </p>
          </div>
        </div>
      </section>

      {/* Playlist - Manage Queue */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <ListMusic size={20} />
          Playlist — Manage Queue
        </h2>
        <p className="text-metal-silver mb-4">
          Build and manage the sequence of content that plays on the display.
        </p>

        <div className="space-y-4">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-1">Drag and Drop Reordering</h3>
            <p className="text-metal-silver text-sm">
              Rearrange items by dragging them to new positions. The currently playing item
              is highlighted, and you can see what's coming up next in the queue.
            </p>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-1">Save and Load Playlists</h3>
            <p className="text-metal-silver text-sm">
              Save your current playlist with a name for later use. Load saved playlists
              to quickly switch between different content sets—one for parties, one for
              ambient display, etc.
            </p>
          </div>

          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-1">Playback Modes</h3>
            <p className="text-metal-silver text-sm">
              Enable auto-repeat to loop the playlist continuously, or shuffle mode to
              randomize the order. Combine both for endless variety.
            </p>
          </div>

          <div className="border-l-2 border-neon-purple pl-4">
            <h3 className="font-retro text-neon-purple mb-1">Transitions</h3>
            <p className="text-metal-silver text-sm">
              Configure how items transition from one to the next. Choose from fade,
              blur, or instant cut. Set transition duration and apply different
              transitions for entering and exiting each item.
            </p>
          </div>
        </div>
      </section>

      {/* Settings - System Config */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Settings size={20} />
          Settings — System Config
        </h2>
        <p className="text-metal-silver mb-4">
          Configure system-wide settings and monitor hardware status.
        </p>

        <div className="space-y-4">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-1">Brightness Control</h3>
            <p className="text-metal-silver text-sm">
              Adjust global LED brightness from 0-100%. Lower brightness extends LED life
              and reduces power consumption—useful for indoor settings or late night use.
            </p>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-1">WiFi Network Management</h3>
            <p className="text-metal-silver text-sm">
              Scan for available networks, connect to new ones, or forget saved networks.
              See signal strength and connection status. If no known network is available,
              the system automatically creates an access point.
            </p>
          </div>

          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-1">Audio Source</h3>
            <p className="text-metal-silver text-sm">
              Switch between microphone input for live audio reactivity, or use a test
              audio file for development and demos. Configure input gain and see real-time
              audio levels.
            </p>
          </div>

          <div className="border-l-2 border-neon-orange pl-4">
            <h3 className="font-retro text-neon-orange mb-1">System Controls</h3>
            <p className="text-metal-silver text-sm">
              Restart the Prismatron service if something goes wrong, or reboot the entire
              system. View current software version and system uptime.
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
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            <span>Install as a PWA on mobile for app-like experience</span>
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
