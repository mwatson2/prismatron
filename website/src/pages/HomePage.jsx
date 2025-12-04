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
        <p className="text-xl md:text-2xl text-neon-pink mb-4">
          A computational LED display where chaos becomes coherent
        </p>
        <p className="text-sm text-metal-silver mb-6 flex items-center justify-center gap-2">
          Made with help from
          <a
            href="https://claude.ai/code"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-neon-orange hover:text-neon-pink transition-colors"
          >
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
              <path d="M4.709 15.955l4.72-2.647.08-.08 2.958-1.665c.084-.05.2-.05.283 0l2.958 1.665.08.08 4.72 2.647c.26.147.26.537 0 .683l-4.72 2.648-.08.079-2.958 1.665c-.084.05-.2.05-.283 0l-2.958-1.665-.08-.08-4.72-2.647c-.26-.146-.26-.536 0-.683zM12 3c.103 0 .205.026.297.078l7.5 4.25c.185.104.297.298.297.508v8.328c0 .21-.112.404-.297.508l-7.5 4.25c-.184.104-.41.104-.594 0l-7.5-4.25C4.018 16.568 3.906 16.374 3.906 16.164V7.836c0-.21.112-.404.297-.508l7.5-4.25C11.795 3.026 11.897 3 12 3z"/>
            </svg>
            Claude Code
          </a>
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
            every single LED, we create a mathematical model of the display. Them, 67Tflops of finest
            Kuang-grade Nvidia silicon runs a real-time optimization
            algorithm that calculates the precise brightness for each LED to best approximate
            any target image or video—turning random chaos into coherent visuals.
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
        <p className="text-metal-silver mb-6">As well as a lot of fun with an exciting result, this project was in large part a learning opportunity:</p>
        <ul className="space-y-2 text-metal-silver">
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">Mechanical Engineering</strong> — nothing I had done before - who knew acrylic could
             be made more rigid by installing with a very slight bow - and how much weight do we need to stop the thing blowing over in a Burning Man dust storm
            ?</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">Electrical / LEDs</strong> — turns out there's a whole world of LED engineering to draw on</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">GPU optimization, modern tensor cores</strong> — did you know headline GPU speeds are valid only if you
            happen to be calculating exactly the right kind of thing ? (basically an AI matrix multiplication)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">Audio-reactive systems</strong> — how do we process audio to extract music features</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">AI assistance and coding</strong> — this project would not have been possible (in the time available) without Claude</span>
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
