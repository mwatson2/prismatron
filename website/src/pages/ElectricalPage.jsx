import { motion } from 'framer-motion'
import { Zap, Lightbulb, Cable } from 'lucide-react'

export default function ElectricalPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <Zap className="text-neon-yellow" />
        Electrical System
      </h1>

      {/* LED Array */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Lightbulb size={20} />
          LED Array
        </h2>

        <div className="bg-dark-800 p-4 rounded border border-neon-cyan/20 mb-6">
          <h4 className="font-retro text-neon-cyan text-sm mb-3">Specifications</h4>
          <ul className="text-metal-silver space-y-1 text-sm">
            <li>• ~3,200 WS2811/WS2812B addressable RGB LEDs</li>
            <li>• Primarily 12V LED string lights with some standard-density LED strips</li>
            <li>• 12V operation reduces voltage drop and power dissipation across long runs</li>
            <li>• Arranged in deliberate randomness—varying density across the display area</li>
          </ul>
        </div>

        <p className="text-metal-silver">
          The LEDs are arranged in organic, curving patterns rather than straight lines.
          String lights provide point sources with natural spacing, while LED strips fill
          denser areas. This intentional irregularity is what gives Prismatron its unique character.
        </p>

        <div className="mt-6 p-4 retro-panel rounded border border-neon-green/30">
          <div className="text-neon-green/50 text-sm font-mono mb-2">[PHOTO PLACEHOLDER]</div>
          <p className="text-metal-silver text-sm">LED arrangement before front panel is attached</p>
        </div>
      </section>

      {/* Power Distribution */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Zap size={20} />
          Power Distribution
        </h2>

        <div className="bg-dark-800 rounded border border-neon-cyan/20 overflow-hidden mb-6">
          <table className="spec-table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Power Draw</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>LED array (typical use)</td>
                <td className="text-neon-cyan">~320W @ 12V</td>
              </tr>
              <tr>
                <td>Jetson Orin Nano</td>
                <td className="text-neon-cyan">~20W @ 19V</td>
              </tr>
              <tr>
                <td>QuinLED DigiOcta controller</td>
                <td className="text-neon-cyan">~5W @ 5V</td>
              </tr>
              <tr className="font-bold bg-dark-700">
                <td>Total</td>
                <td className="text-neon-pink">~345W</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p className="text-metal-silver mb-4">
          The 12V LED system significantly reduces current requirements compared to 5V.
          At 12V, the same power is delivered with 1/2.4 the current, dramatically reducing
          voltage drop across long wire runs and power dissipation in the conductors.
        </p>

        <p className="text-metal-silver">
          Power injection is provided at the start and end of each LED string/strip.
          The Jetson and LED controller have separate, isolated power supplies to prevent
          noise interference.
        </p>
      </section>

      {/* Control Signal Path */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Cable size={20} />
          Control Signal Path
        </h2>

        <div className="diagram-box mb-6">
{`[Jetson Orin Nano] --UDP/DDP--> [QuinLED DigiOcta] --Data--> [LED Strings]
                     (WiFi)         (WLED firmware)      (8 parallel outputs)`}
        </div>

        <p className="text-metal-silver mb-4">
          The Jetson calculates LED values and sends them over WiFi using the DDP
          (Distributed Display Protocol)—a simple UDP-based protocol supported by WLED.
          This eliminates the need for wired connections between the compute unit and
          LED controller.
        </p>

        <p className="text-metal-silver">
          The QuinLED DigiOcta runs WLED firmware and drives 8 parallel LED data outputs,
          enabling fast refresh rates across all 3,200 LEDs. Each output handles a segment
          of the display, with the controller managing timing-critical LED protocol generation.
        </p>

        <div className="mt-6 p-4 retro-panel rounded border border-neon-purple/30">
          <div className="text-neon-purple/50 text-sm font-mono mb-2">[PHOTO PLACEHOLDER]</div>
          <p className="text-metal-silver text-sm">QuinLED DigiOcta and wiring harness</p>
        </div>
      </section>
    </motion.div>
  )
}
