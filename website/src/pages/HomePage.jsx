import { motion } from 'framer-motion'
import { ArrowRight, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import heroVideo from '../media/PrismatronExample2.mp4'
import secondaryVideo from '../media/PrismatronExample1.mp4'
import Comments from '../components/Comments'

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

        <p className="text-metal-silver max-w-2xl mx-auto text-lg mb-6 leading-relaxed">
          3,200 LEDs. Randomly arranged. Mathematically orchestrated. Prismatron transforms
          scattered points of light into recognizable images through real-time computational
          optimization—proving that order can emerge from apparent disorder.
        </p>
        <p className="text-metal-silver/80 max-w-2xl mx-auto mb-6 leading-relaxed">
          Prismatron was created by Mark Watson, but several people and some non-people helped. Thanks and love
          to my family for their extreme patience with the many nights and weekends this project consumed.
          Thanks to Paul J for sterling logistical support getting the thing to and from Burning Man.
          Finally, thanks to (the creators of){' '}
          <a
            href="https://claude.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-neon-orange hover:text-neon-pink transition-colors"
          >
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
              <path d="M4.709 15.955l4.72-2.647.08-.08 2.958-1.665c.084-.05.2-.05.283 0l2.958 1.665.08.08 4.72 2.647c.26.147.26.537 0 .683l-4.72 2.648-.08.079-2.958 1.665c-.084.05-.2.05-.283 0l-2.958-1.665-.08-.08-4.72-2.647c-.26-.146-.26-.536 0-.683zM12 3c.103 0 .205.026.297.078l7.5 4.25c.185.104.297.298.297.508v8.328c0 .21-.112.404-.297.508l-7.5 4.25c-.184.104-.41.104-.594 0l-7.5-4.25C4.018 16.568 3.906 16.374 3.906 16.164V7.836c0-.21.112-.404.297-.508l7.5-4.25C11.795 3.026 11.897 3 12 3z"/>
            </svg>
            Claude
          </a>{' '}and{' '}
          <a
            href="https://claude.ai/code"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-neon-orange hover:text-neon-pink transition-colors"
          >
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
              <path d="M4.709 15.955l4.72-2.647.08-.08 2.958-1.665c.084-.05.2-.05.283 0l2.958 1.665.08.08 4.72 2.647c.26.147.26.537 0 .683l-4.72 2.648-.08.079-2.958 1.665c-.084.05-.2.05-.283 0l-2.958-1.665-.08-.08-4.72-2.647c-.26-.146-.26-.536 0-.683zM12 3c.103 0 .205.026.297.078l7.5 4.25c.185.104.297.298.297.508v8.328c0 .21-.112.404-.297.508l-7.5 4.25c-.184.104-.41.104-.594 0l-7.5-4.25C4.018 16.568 3.906 16.374 3.906 16.164V7.836c0-.21.112-.404.297-.508l7.5-4.25C11.795 3.026 11.897 3 12 3z"/>
            </svg>
            Claude Code
          </a>, without which this project would not have been possible.
        </p>

        <div className="mt-10 retro-panel rounded-lg border border-neon-cyan/50 overflow-hidden">
          <video
            src={heroVideo}
            autoPlay
            loop
            muted
            playsInline
            className="w-full max-w-4xl mx-auto"
          />
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
            <span><strong className="text-neon-cyan">AI assistance and coding</strong> — this project would not have been possible (in the time available) without Claude.
            Working on a green-field project in a new subject-matter area gave me a chance to develop a great sense of the strengths and limits of this tool.</span>
          </li><li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">Mechanical Engineering</strong> — I would not have known where to start to calculate the thickness of aluminum needed,
            the number of supports, how much weight we need to stop the thing blowing over in a Burning Man dust storm and much much more ...
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">Electrical / LEDs</strong> — turns out there's a whole world of LED engineering to draw on</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">GPU optimization, modern tensor cores</strong> — did you know headline GPU speeds are valid only if you
            happen to be calculating exactly the right kind of thing ? If you're not doing the kind of matrix-matrix multiplication that is used in AI workflows
            you're stuck with the old CUDA cores with much lower capacity. So then wrangling the problem in to the right shape was fun.</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan">•</span>
            <span><strong className="text-neon-cyan">Audio-reactive systems</strong> — how do we process audio to extract music features - surprisingly
            you can get a long way just by reversing the polarity of the neutron flux ... no, sorry, wrong kind of flux ... it's <em>spectral</em> flux and it's
            slope is very informative as to what is going on in the music</span>
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
          { path: '/compute', title: 'Compute Platform', desc: 'Jetson Orin Nano and LED driver' },
          { path: '/algorithm', title: 'The Algorithm', desc: 'Calibration and optimization' },
          { path: '/audio', title: 'Audio-Reactive', desc: 'Beyond beat detection' },
          { path: '/software', title: 'Software Architecture', desc: 'System components and design' },
          { path: '/control-app', title: 'Control App', desc: 'Web-based interface' },
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

      {/* Secondary Video */}
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="mt-10 retro-panel rounded-lg border border-neon-cyan/50 overflow-hidden"
      >
        <video
          src={secondaryVideo}
          autoPlay
          loop
          muted
          playsInline
          className="w-full max-w-4xl mx-auto"
        />
      </motion.section>

      <Comments pageSlug="home" />
    </div>
  )
}
