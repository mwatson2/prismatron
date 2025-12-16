import { motion } from 'framer-motion'
import { Box, Thermometer, Move } from 'lucide-react'
import Comments from '../components/Comments'

export default function MechanicalPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <Box className="text-neon-pink" />
        Mechanical Design
      </h1>

      {/* Sandwich Structure */}
      <section className="content-card">
        <h2 className="subsection-title">The Sandwich Structure</h2>
        <p className="text-metal-silver mb-6">
          Prismatron uses a dual-panel design with a 4-inch air gap between layers:
        </p>

        <div className="space-y-6">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-2">Front Panel — The Diffuser</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• 1/4" textured acrylic sheet (60" × 36")</li>
              <li>• Creates unique, irregular light patterns from each LED</li>
              <li>• Mounted with cylindrical aluminium standoffs close to the edges</li>
              <li>• The texture transforms point sources into sparkling light shapes</li>
            </ul>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-2">Back Panel — The Heat Sink</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• 3mm 6061 aluminum sheet, CNC rounded corners and mounting holes</li>
              <li>• Provides structural rigidity and thermal management</li>
              <li>• Dissipates heat from 3,200 LEDs without requiring active cooling</li>
            </ul>
          </div>

          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-2">The Gap</h3>
            <ul className="text-metal-silver space-y-1 text-sm">
              <li>• 4 inches of separation allows light to spread before hitting the diffuser</li>
              <li>• Creates space for LED strips and wiring</li>
              <li>• Enables passive airflow for cooling</li>
            </ul>
          </div>
        </div>

      </section>

      {/* Mounting System */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Move size={20} />
          Mounting System
        </h2>
        <p className="text-metal-silver mb-4">
          The back panel attaches to a standard VESA TV mount, making Prismatron easy to hang
          on walls or mount on stands. The front panel floats in front, held by cylindrical post
          standoffs. We used an off-the-shelf TV Easel mount to install in any enviroment.
        </p>

        <div className="bg-dark-800 p-4 rounded border border-neon-cyan/20">
          <h4 className="font-retro text-neon-cyan text-sm mb-3">Weight Breakdown</h4>
          <table className="w-full text-sm">
            <tbody className="text-metal-silver">
              <tr className="border-b border-dark-600">
                <td className="py-2">Front acrylic panel</td>
                <td className="py-2 text-right text-neon-cyan">~23 lbs</td>
              </tr>
              <tr className="border-b border-dark-600">
                <td className="py-2">Back aluminum panel</td>
                <td className="py-2 text-right text-neon-cyan">~25 lbs</td>
              </tr>
              <tr className="border-b border-dark-600">
                <td className="py-2">LEDs, wiring, electronics</td>
                <td className="py-2 text-right text-neon-cyan">~5 lbs</td>
              </tr>
              <tr className="font-bold">
                <td className="py-2">Total</td>
                <td className="py-2 text-right text-neon-pink">~53 lbs</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Thermal Management */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Thermometer size={20} />
          Thermal Management
        </h2>
        <p className="text-metal-silver mb-4">
          With 3,200 LEDs potentially drawing 320W at full brightness, heat management is
          critical—especially for outdoor deployment in desert conditions (30-35°C ambient).
        </p>
        <p className="text-metal-silver mb-4">
          The aluminum back panel serves as a passive heat sink. Its thermal conductivity
          (167 W/m·K for 6061 alloy) is nearly 1,000 times higher than acrylic, efficiently
          spreading and radiating heat away from the LEDs.
        </p>
        <p className="text-metal-silver">
          The 4-inch air gap enables natural convection, and the open edges allow continuous
          airflow. <span className="text-neon-green">No fans required.</span>
        </p>
      </section>

      <Comments pageSlug="mechanical" />
    </motion.div>
  )
}
