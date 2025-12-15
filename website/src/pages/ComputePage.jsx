import { motion } from 'framer-motion'
import { Cpu, Wifi } from 'lucide-react'

export default function ComputePage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <Cpu className="text-neon-green" />
        Compute Platform
      </h1>

      {/* Why a Jetson */}
      <section className="content-card">
        <h2 className="subsection-title">Why a Jetson?</h2>
        <p className="text-metal-silver mb-4">
          The core challenge of Prismatron is solving an optimization problem in real-time:
          given a target image, find the brightness values for 3,200 LEDs that best approximate
          it when their diffused light patterns combine.
        </p>
        <p className="text-metal-silver mb-6">
          This is computationally intensive.
          A Jetson Orin Nano provides:
        </p>

        <ul className="space-y-3 text-metal-silver">
          <li className="flex items-start gap-3">
            <span className="text-neon-green font-bold">•</span>
            <span><strong className="text-neon-green">40-67 TOPS</strong> of AI/compute performance</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="text-neon-green font-bold">•</span>
            <span><strong className="text-neon-green">GPU acceleration</strong> for parallel matrix operations</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="text-neon-green font-bold">•</span>
            <span><strong className="text-neon-green">Low power</strong> (~15-20W) for portable deployment</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="text-neon-green font-bold">•</span>
            <span><strong className="text-neon-green">Compact size</strong> that fits within the display assembly</span>
          </li>
        </ul>

        <p className="text-metal-silver mt-6">
          For context, this is roughly <span className="text-neon-pink">80x more powerful</span> than
          the original Jetson Nano, enabling real-time optimization that would otherwise require
          a desktop workstation.
        </p>

        <div className="mt-6 p-4 retro-panel rounded border border-neon-green/30">
          <div className="text-neon-green/50 text-sm font-mono mb-2">[PHOTO PLACEHOLDER]</div>
          <p className="text-metal-silver text-sm">Jetson Orin Nano mounted in the assembly</p>
        </div>
      </section>

      {/* QuinLED DigiOcta as LED Driver */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Wifi size={20} />
          QuinLED DigiOcta as LED Driver
        </h2>
        <p className="text-metal-silver mb-4">
          The QuinLED DigiOcta running WLED firmware handles the "last mile" of LED control:
        </p>

        <ul className="space-y-2 text-metal-silver">
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            Receives computed brightness values from the Jetson via WiFi (UDP/DDP protocol)
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            Converts data to the precise timing protocol WS2811/WS2812B LEDs require
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            Drives 8 parallel LED data outputs for fast refresh across all segments
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">→</span>
            WLED firmware provides web configuration, effects fallback, and diagnostics
          </li>
        </ul>

        <p className="text-metal-silver mt-6">
          This division of labor lets each processor do what it's best at: the Jetson handles
          heavy computation and optimization, while the DigiOcta handles real-time LED protocol
          generation with microsecond-precise timing.
        </p>
      </section>

      {/* System Architecture */}
      <section className="content-card">
        <h2 className="subsection-title">System Architecture</h2>
        <div className="diagram-box text-xs">
{`┌─────────────────────────────────────────────────────────┐
│                    Jetson Orin Nano                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Audio Input │  │ Image Input │  │  Mobile App API │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         │                │                   │          │
│         ▼                ▼                   ▼          │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Image producer pipeline                ││
│  │            (video, still image, effects)            ││
│  └──────────────────────┬──────────────────────────────┘│
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────────┐│
│  │           Optimization Engine (GPU)                 ││
│  │     LED patterns × target → brightness values       ││
│  └──────────────────────┬──────────────────────────────┘│
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────────┐│
│  │                   Renderer.                         ││
│  │     Audio-reactive and other render-time effects    ││
│  └──────────────────────┬──────────────────────────────┘│
│                         │                               │
│                         ▼ UDP/DDP (WiFi)                │
└─────────────────────────┼───────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  QuinLED DigiOcta     │
              │  (WLED firmware)      │
              │  8 parallel outputs   │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    3,200 RGB LEDs     │
              │  (12V string lights)  │
              └───────────────────────┘`}
        </div>
      </section>
    </motion.div>
  )
}
