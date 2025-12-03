import { motion } from 'framer-motion'
import { ArrowRight, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'

export default function HomePage() {
  return (
    <div>
      {/* Hero Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center py-12 lg:py-20"
      >
        <h1 className="font-retro text-4xl md:text-6xl text-neon-cyan text-neon-strong mb-4 animate-glow">
          PRISMATRON
        </h1>
        <p className="text-xl md:text-2xl text-neon-pink mb-6">
          A computational LED display where chaos becomes coherent
        </p>
        <p className="text-metal-silver max-w-2xl mx-auto text-lg leading-relaxed">
          3,200 LEDs. Randomly arranged. Mathematically orchestrated. Prismatron transforms
          scattered points of light into recognizable images through real-time computational
          optimization—proving that order can emerge from apparent disorder.
        </p>

        <div className="mt-10 p-8 retro-panel rounded-lg border border-neon-cyan/50">
          <div className="text-neon-cyan/50 text-sm font-mono mb-2">[HERO VIDEO PLACEHOLDER]</div>
          <p className="text-metal-silver text-sm">
            Slow pan across the display showing an image forming from individual LED points
          </p>
        </div>
      </motion.section>

      {/* What is Prismatron */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="content-card"
      >
        <h2 className="section-title flex items-center gap-3">
          <Sparkles className="text-neon-pink" />
          What is Prismatron?
        </h2>
        <div className="space-y-4 text-metal-silver leading-relaxed">
          <p>
            Prismatron is an experimental LED art installation that takes a counterintuitive
            approach to display technology. Rather than arranging LEDs in a precise grid like
            a traditional screen, Prismatron scatters 3,200 RGB LEDs in deliberately haphazard
            patterns between two panels. The front panel's textured surface diffuses each LED
            into a unique, irregular light shape.
          </p>
          <p>
            The magic happens in software. By photographing the exact diffusion pattern of
            every single LED, we create a mathematical model of the display. An optimization
            algorithm then calculates the precise brightness for each LED to best approximate
            any target image—turning random chaos into coherent visuals.
          </p>
          <p>
            The result is something that feels alive. Images emerge from the noise with an
            organic, almost dreamlike quality that grid-based displays can't replicate.
          </p>
        </div>
      </motion.section>

      {/* Why Build This */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="content-card"
      >
        <h2 className="subsection-title">Why Build This?</h2>
        <p className="text-metal-silver mb-6">
          Traditional LED displays optimize for pixel density and color accuracy. Prismatron
          asks a different question: what if we embraced imperfection and let computation
          bridge the gap?
        </p>
        <p className="text-metal-silver mb-4">This project explores the intersection of:</p>
        <ul className="space-y-2 text-metal-silver">
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">Computational photography</strong> — using calibration to understand how physical systems behave</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">Optimization theory</strong> — finding the best solution among billions of possibilities</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">Generative art</strong> — creating visuals with emergent, organic qualities</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">Audio-reactive systems</strong> — responding to music in real-time</span>
          </li>
        </ul>
        <p className="text-metal-silver mt-6">
          Built for environments like Burning Man, Prismatron is designed to be rugged,
          portable, and mesmerizing.
        </p>
      </motion.section>

      {/* Navigation Cards */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="grid md:grid-cols-2 gap-4 mt-8"
      >
        {[
          { path: '/mechanical', title: 'Mechanical Design', desc: 'Panel structure and thermal management' },
          { path: '/electrical', title: 'Electrical System', desc: 'LED array and power distribution' },
          { path: '/algorithm', title: 'The Algorithm', desc: 'Calibration and optimization' },
          { path: '/audio', title: 'Audio-Reactive', desc: 'Beyond beat detection' },
        ].map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className="retro-panel p-5 rounded-lg hover:border-neon-pink/50 transition-all group"
          >
            <h3 className="font-retro text-neon-cyan group-hover:text-neon-pink transition-colors flex items-center gap-2">
              {item.title}
              <ArrowRight size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
            </h3>
            <p className="text-metal-silver text-sm mt-1">{item.desc}</p>
          </Link>
        ))}
      </motion.section>
    </div>
  )
}
