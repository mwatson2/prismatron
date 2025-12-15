import { motion } from 'framer-motion'
import { Flame, Wind, Shield, Anchor } from 'lucide-react'

export default function BurningManPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <Flame className="text-neon-orange" />
        Burning Man 2025
      </h1>

      {/* Hero Video */}
      <section className="content-card">
        <div className="retro-panel rounded-lg border border-neon-orange/50 overflow-hidden mb-6">
          <video
            src="/media/PrismatronBurningMan.mp4"
            autoPlay
            loop
            muted
            playsInline
            controls
            className="w-full"
          />
        </div>
        <p className="text-metal-silver">
          The Prismatron made its debut at Burning Man 2025! Generously hosted by the{' '}
          <span className="text-neon-orange">Black Rock Sauna Society</span>, the display
          provided evening light shows at our Black Rock City corner at 4:30 & E.
        </p>
        <p className="text-metal-silver mt-4">
          An unfortunate 10-bit vs 8-bit color depth mixup in the video pipeline meant we
          had no video support for the burn—but we made the most of it with scrolling text
          displays and procedural effects. Sometimes constraints breed creativity.
        </p>
      </section>

      {/* Dust Protection */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Wind size={20} />
          Surviving the Playa Dust
        </h2>
        <p className="text-metal-silver mb-4">
          The Black Rock Desert environment is notoriously hostile to electronics. The
          alkaline playa dust is incredibly fine—particles as small as 0.5 microns—and
          gets into everything. It's conductive when mixed with moisture, corrosive to
          metals, and abrasive to moving parts. Standard consumer electronics rarely
          survive a week on the playa without protection.
        </p>

        <div className="space-y-4">
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-2">Jetson Enclosure</h3>
            <p className="text-metal-silver text-sm">
              To protect the Jetson Orin Nano, I built a secondary outer enclosure with
              minimal openings. Air intake and exhaust apertures were fitted with fine
              mesh filters, and small fans maintained positive pressure inside to prevent
              dust infiltration. The enclosure kept the compute unit running flawlessly
              throughout the event.
            </p>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-2">Panel Protection</h3>
            <p className="text-metal-silver text-sm">
              The open edges between the front and back panels, and between the front
              and back planes of the electronics panel, were protected with 80-mesh
              stainless steel mesh. At roughly 180 microns, this isn't anywhere close
              to fine enough to stop microscopic playa dust—but it keeps out larger
              debris and provides some wind break for the finer particles.
            </p>
          </div>

          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-2">Dust Storm Protocol</h3>
            <p className="text-metal-silver text-sm">
              For dust storms and whiteouts, an off-the-shelf outdoor TV cover provided
              full enclosure protection. Quick to deploy, it kept the worst conditions
              at bay while we waited for visibility to return.
            </p>
          </div>
        </div>
      </section>

      {/* Thermal Considerations */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Shield size={20} />
          Thermal Management on the Playa
        </h2>
        <p className="text-metal-silver mb-4">
          Heat was originally a major concern. Daytime temperatures on the playa can
          exceed 100°F (38°C), and direct sun exposure can push surface temperatures
          much higher. Running 320W of LEDs plus compute in those conditions seemed
          like a recipe for thermal throttling—or worse.
        </p>
        <p className="text-metal-silver">
          Then I realized: <span className="text-neon-green">the display would basically
          only be used at night!</span> By the time the Prismatron powered on each evening,
          ambient temperatures had dropped significantly. The aluminum back panel's
          passive cooling proved more than adequate, and we never encountered thermal
          issues during operation.
        </p>
      </section>

      {/* Stability */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Anchor size={20} />
          Keeping It Upright
        </h2>
        <p className="text-metal-silver mb-4">
          Mounted on a TV easel stand, the Prismatron is decidedly top-heavy. The ~53 lb
          display sitting high on a lightweight stand made for a precarious setup—one
          strong gust or stumbling burner away from disaster.
        </p>

        <div className="bg-dark-800 p-4 rounded border border-neon-cyan/20 mb-4">
          <h4 className="font-retro text-neon-cyan text-sm mb-3">The Anchor Solution</h4>
          <ul className="text-metal-silver space-y-1 text-sm">
            <li>• Sourced a 4' × 1' steel plate offcut from a local metal supplier</li>
            <li>• Plate weighs approximately 60 lbs—more than the display itself</li>
            <li>• Connected to the easel base with 1" square aluminum tubes</li>
            <li>• Low center of gravity dramatically improved stability</li>
          </ul>
        </div>

        <p className="text-metal-silver">
          The steel anchor wouldn't have survived a direct hit from a full-force playa
          storm—but within the relative shelter of Black Rock City's street grid, it
          proved plenty secure. The display survived the week without incident, even
          through several moderate wind events.
        </p>
      </section>
    </motion.div>
  )
}
