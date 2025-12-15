import { motion } from 'framer-motion'
import { Music, Waves, Palette } from 'lucide-react'

export default function AudioPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <Music className="text-neon-pink" />
        Audio-Reactive Mode
      </h1>

      {/* Beyond Beat Detection */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Waves size={20} />
          Beyond Beat Detection
        </h2>
        <p className="text-metal-silver mb-4">
          Prismatron doesn't just flash to the beat—it understands musical structure.
          Using the Aubio library for real-time audio analysis, the system detects:
        </p>

        <ul className="space-y-2 text-metal-silver">
          <li className="flex items-start gap-2">
            <span className="text-neon-pink">•</span>
            <span><strong className="text-neon-pink">Tempo and beats</strong> — The fundamental rhythm</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-pink">•</span>
            <span><strong className="text-neon-pink">Spectral content</strong> — Which frequencies are present and how loud</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-pink">•</span>
            <span><strong className="text-neon-pink">Transients</strong> — Sudden changes like drum hits</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-pink">•</span>
            <span><strong className="text-neon-pink">Musical patterns</strong> — Build-ups, cuts, and drops</span>
          </li>
        </ul>

        <p className="text-metal-silver mt-6">
          For electronic dance music, this means recognizing when snare rolls are building tension,
          detecting the "cut" when energy suddenly drops, and anticipating the bass return on the drop.
          The visuals follow the emotional arc of the music—not just its volume.
        </p>

        <div className="mt-6 p-4 retro-panel rounded border border-neon-pink/30">
          <div className="text-neon-pink/50 text-sm font-mono mb-2">[VIDEO PLACEHOLDER]</div>
          <p className="text-metal-silver text-sm">Prismatron responding to EDM track, showing build-up/drop response</p>
        </div>
      </section>

      {/* Visual Mapping */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Palette size={20} />
          Visual Mapping
        </h2>
        <p className="text-metal-silver mb-6">
          Audio features map to visual parameters:
        </p>

        <div className="bg-dark-800 rounded border border-neon-cyan/20 overflow-hidden">
          <table className="spec-table">
            <thead>
              <tr>
                <th>Audio Feature</th>
                <th>Visual Response</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="text-neon-cyan">Beat</td>
                <td>Pulse/flash intensity / Trigger template animation</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">Bass energy</td>
                <td>Color warmth, overall brightness (not implemeted yet)</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">Build-up detection</td>
                <td>Sparkle effect with increasing intensity</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">Cut before the drop</td>
                <td>Drop to black and fade back</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">Drop</td>
                <td>Full-field max brightness and fade back to image</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p className="text-metal-silver mt-6">
          The mapping is configurable, allowing experimentation and customization.
        </p>
      </section>

      {/* Build-up/Drop Detection */}
      <section className="content-card">
        <h2 className="subsection-title">Build-Up/Cut/Drop Detection</h2>
        <p className="text-metal-silver mb-4">
          The system uses spectral analysis to detect musical energy patterns common in
          house and trance music:
        </p>

        <div className="space-y-4 mb-6">
          <div className="border-l-2 border-neon-yellow pl-4">
            <h3 className="font-retro text-neon-yellow mb-2">Build-Up Detection</h3>
            <p className="text-metal-silver text-sm">
              Triggered by <strong>snare rolls</strong> which we detected via autocorrelation in the frequency bands
              associated with snare drums, looking specifically for autocorrelation at 4x or 8x BPM
              We also look for <strong>rising high-frequency flux</strong> and <strong>rising spectral centroid</strong>.
              Build-up intensity is a continuous value that increases as the build progresses.
            </p>
          </div>

          <div className="border-l-2 border-neon-orange pl-4">
            <h3 className="font-retro text-neon-orange mb-2">Cut Detection</h3>
            <p className="text-metal-silver text-sm">
              Detected when <strong>mid-range energy suddenly drops</strong> below 50% of the long-term
              average during a build-up. This is the classic "everything cuts out" moment before the drop.
            </p>
          </div>

          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-2">Drop Detection</h3>
            <p className="text-metal-silver text-sm">
              Triggered by a <strong>sudden bass flux spike</strong> either during a build-up or within
              2 bars after a cut. This is the moment the bass returns with full force.
            </p>
          </div>
        </div>

        <h3 className="font-retro text-neon-cyan text-sm mb-3">Detection Signals</h3>
        <div className="grid gap-2 text-sm">
          <div className="flex items-center gap-3 p-3 bg-dark-800 rounded">
            <span className="text-neon-cyan font-mono text-xs w-32">Spectral Centroid</span>
            <span className="text-dark-300">Rising = brighter timbres (hi-hats, synth builds)</span>
          </div>
          <div className="flex items-center gap-3 p-3 bg-dark-800 rounded">
            <span className="text-neon-cyan font-mono text-xs w-32">High/Air Flux</span>
            <span className="text-dark-300">Rate of change in 2-16 kHz range</span>
          </div>
          <div className="flex items-center gap-3 p-3 bg-dark-800 rounded">
            <span className="text-neon-cyan font-mono text-xs w-32">Bass Flux</span>
            <span className="text-dark-300">Rate of change in 20-250 Hz range</span>
          </div>
          <div className="flex items-center gap-3 p-3 bg-dark-800 rounded">
            <span className="text-neon-cyan font-mono text-xs w-32">Mid Energy Ratio</span>
            <span className="text-dark-300">Short-term vs long-term mid-range power</span>
          </div>
          <div className="flex items-center gap-3 p-3 bg-dark-800 rounded">
            <span className="text-neon-cyan font-mono text-xs w-32">Snare Roll</span>
            <span className="text-dark-300">Autocorrelation peak at 4x/8x BPM in snare bands</span>
          </div>
        </div>
      </section>

      {/* Technical Details */}
      <section className="content-card">
        <h2 className="subsection-title">Technical Details</h2>
        <p className="text-metal-silver mb-4">
          The audio analyzer runs at ~86 frames per second (512 sample hop size at 44.1 kHz),
          using EWMA (exponentially-weighted moving average) filters to smooth signals
          while maintaining responsiveness:
        </p>

        <ul className="space-y-2 text-metal-silver text-sm">
          <li className="flex items-start gap-2">
            <span className="text-neon-green">•</span>
            <span><strong className="text-neon-green">Spectral centroid:</strong> 1s input smoothing → 2s slope smoothing</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-green">•</span>
            <span><strong className="text-neon-green">Flux signals:</strong> 0.5s input smoothing → 1s slope smoothing</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-green">•</span>
            <span><strong className="text-neon-green">Mid energy:</strong> 0.25s short-term vs 4s long-term for ratio</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-green">•</span>
            <span><strong className="text-neon-green">Snare detection:</strong> ~1s autocorrelation window</span>
          </li>
        </ul>

        <p className="text-metal-silver mt-4">
          Slopes are calculated over 0.25s intervals to detect rising/falling trends rather than
          instantaneous values, making detection robust to momentary fluctuations.
        </p>
        <p className="text-metal-silver mt-4">
          We also implemented automatic gain control to handle varying input volumes together with a slight bass boost
          to compensate for poor bass response in the (very cheap!) microphone used. A high-pass filter is used
          to remove very low-frequence noise.
        </p>
        <p className="text-metal-silver mt-4">
          For development / tuning, the system can be set to use a wav file input on a loop. This allows experimentation
          with a known input signal. We also provide an option to capture and output the microphone signal, allowing
          offline experimentation with the signal exactly as it is heard by the system.
        </p>
      </section>
    </motion.div>
  )
}
