import { motion } from 'framer-motion'
import { FileText } from 'lucide-react'

export default function SpecsPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <FileText className="text-neon-orange" />
        Technical Specifications
      </h1>

      <section className="content-card">
        <div className="bg-dark-800 rounded border border-neon-cyan/20 overflow-hidden">
          <table className="spec-table">
            <thead>
              <tr>
                <th>Specification</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Display dimensions</td>
                <td className="text-neon-cyan">60" × 36" (152 × 91 cm)</td>
              </tr>
              <tr>
                <td>LED count</td>
                <td className="text-neon-cyan">~3,200 RGB</td>
              </tr>
              <tr>
                <td>LED type</td>
                <td className="text-neon-cyan">WS2811/WS2812B (12V string lights + strips)</td>
              </tr>
              <tr>
                <td>LED voltage</td>
                <td className="text-neon-cyan">12V DC</td>
              </tr>
              <tr>
                <td>Panel gap</td>
                <td className="text-neon-cyan">4 inches (10 cm)</td>
              </tr>
              <tr>
                <td>Front panel</td>
                <td className="text-neon-cyan">1/4" textured acrylic</td>
              </tr>
              <tr>
                <td>Back panel</td>
                <td className="text-neon-cyan">3mm 6061 aluminum</td>
              </tr>
              <tr>
                <td>Total weight</td>
                <td className="text-neon-cyan">~53 lbs (24 kg)</td>
              </tr>
              <tr>
                <td>Power consumption</td>
                <td className="text-neon-cyan">~345W typical</td>
              </tr>
              <tr>
                <td>Frame rate</td>
                <td className="text-neon-cyan">~30+ fps (dynamic)</td>
              </tr>
              <tr>
                <td>Compute platform</td>
                <td className="text-neon-cyan">NVIDIA Jetson Orin Nano 8GB</td>
              </tr>
              <tr>
                <td>LED controller</td>
                <td className="text-neon-cyan">QuinLED DigiOcta (WLED firmware)</td>
              </tr>
              <tr>
                <td>Communication</td>
                <td className="text-neon-cyan">UDP/DDP over WiFi (Jetson → DigiOcta)</td>
              </tr>
              <tr>
                <td>Power injection</td>
                <td className="text-neon-cyan">Start and end of each strip/string</td>
              </tr>
              <tr>
                <td>Operating temperature</td>
                <td className="text-neon-cyan">0-45°C ambient</td>
              </tr>
              <tr>
                <td>Mounting</td>
                <td className="text-neon-cyan">VESA compatible</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* About */}
      <section className="content-card mt-8">
        <h2 className="subsection-title">About</h2>
        <p className="text-metal-silver mb-4">
          Prismatron is a personal project exploring the intersection of computational imaging,
          optimization algorithms, and LED art. It's designed for desert festivals, gallery
          installations, and anywhere that could use a conversation-starting light display.
        </p>
        <p className="text-dark-300 text-sm italic mt-6">
          Built with passion for the playa.
        </p>
      </section>

      {/* Footer */}
      <div className="mt-12 text-center py-8 border-t border-neon-cyan/20">
        <p className="font-retro text-neon-cyan text-neon animate-pulse-neon">
          Where 3,200 points of light learn to paint
        </p>
      </div>
    </motion.div>
  )
}
