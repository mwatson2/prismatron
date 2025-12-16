import { motion } from 'framer-motion'
import { Code, Camera, Gauge, Zap, Cpu } from 'lucide-react'
import Comments from '../components/Comments'

export default function AlgorithmPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <Code className="text-neon-purple" />
        The Algorithm
      </h1>

      {/* Mathematical Problem Setup */}
      <section className="content-card">
        <h2 className="subsection-title">The Optimization Problem</h2>
        <p className="text-metal-silver mb-4">
          Prismatron's core challenge is finding the right brightness for each LED so that
          when their diffused light patterns combine, the result approximates a target image.
          This is a <strong className="text-neon-cyan">linear least-squares problem</strong>.
        </p>

        <div className="bg-dark-800 p-4 rounded border border-neon-purple/30 mb-6">
          <p className="text-metal-silver mb-3">Given:</p>
          <ul className="space-y-2 text-metal-silver text-sm mb-4">
            <li><span className="text-neon-cyan font-mono">A</span> — The diffusion pattern matrix (pixels × LEDs)</li>
            <li><span className="text-neon-cyan font-mono">b</span> — The target image (pixels × 1)</li>
            <li><span className="text-neon-cyan font-mono">x</span> — LED brightness values to solve for (LEDs × 1)</li>
          </ul>
          <p className="text-metal-silver">We want to find <span className="text-neon-cyan font-mono">x</span> that minimizes:</p>
          <div className="text-center my-4 font-mono text-neon-pink text-lg">
            ‖Ax - b‖²
          </div>
          <p className="text-metal-silver text-sm">
            Subject to the constraint: <span className="text-neon-green font-mono">0 ≤ x ≤ 1</span> (normalized brightness)
          </p>
        </div>

        <p className="text-metal-silver">
          Each column of <span className="font-mono text-neon-cyan">A</span> represents one LED's diffusion
          pattern—how light from that LED spreads across the image plane when photographed through the
          textured diffuser. With 3,200 LEDs and ~380,000 pixels, the full matrix would be enormous,
          but the diffusion patterns are localized, making the matrix highly sparse. I created a custom matrix
          format with GPU kernels to efficiently store and operate on this matrix. This recognizes that
          the non-zero pixels in each image (one per LED) are localized and so we store only a 64x64 region
          centered on the LED's position. This may involve some cropping if an LED casts a wide image,
          but this is an acceptable trade-off for efficiency.
        </p>
      </section>

      {/* Closed-Form Solution */}
      <section className="content-card">
        <h2 className="subsection-title">The Closed-Form Solution</h2>
        <p className="text-metal-silver mb-4">
          Without constraints, the least-squares problem has an analytical solution using the
          <strong className="text-neon-cyan"> Moore-Penrose pseudo-inverse</strong>:
        </p>

        <div className="bg-dark-800 p-4 rounded border border-neon-cyan/30 mb-6">
          <div className="text-center font-mono text-neon-cyan text-lg mb-3">
            x = (AᵀA)⁻¹ Aᵀb
          </div>
          <p className="text-metal-silver text-sm text-center">
            Where <span className="font-mono">AᵀA</span> is the Gram matrix (LEDs × LEDs)
          </p>
        </div>

        <p className="text-metal-silver mb-4">
          We precompute both <span className="font-mono text-neon-cyan">(AᵀA)</span> and
          <span className="font-mono text-neon-cyan">(AᵀA)⁻¹</span> once during calibration.
          For each new target frame, we only need to compute:
        </p>

        <ol className="space-y-2 text-metal-silver mb-4">
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan font-retro">1.</span>
            <span><span className="font-mono text-neon-pink">Aᵀb</span> — Project target image into LED space</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-neon-cyan font-retro">2.</span>
            <span><span className="font-mono text-neon-pink">(AᵀA)⁻¹ · (Aᵀb)</span> — Apply precomputed inverse</span>
          </li>
        </ol>

        <p className="text-metal-silver mb-4">
          Our custom format for <span className="font-mono text-neon-pink">A</span> makes the computation
          of <span className="font-mono text-neon-pink">Aᵀb</span> very efficient: for each LED we only need to
          calculate the dot product with the corresponding 64x64 region of <span className="font-mono text-neon-pink">b</span>.
        </p>

        <div className="mt-6 p-4 bg-dark-700 rounded border-l-4 border-neon-orange">
          <h4 className="font-retro text-neon-orange text-sm mb-2">The Problem</h4>
          <p className="text-metal-silver text-sm">
            The pseudo-inverse solution doesn't respect our brightness constraints.
            It can produce values <span className="text-neon-pink">x &lt; 0</span> or{' '}
            <span className="text-neon-pink">x &gt; 1</span>, which are physically impossible—LEDs
            can't have negative brightness or exceed their maximum output.
          </p>
        </div>
      </section>

      {/* Constrained Optimization */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Gauge size={20} />
          Projected Gradient Descent
        </h2>
        <p className="text-metal-silver mb-4">
          To enforce the <span className="font-mono text-neon-green">[0, 1]</span> constraint, we use
          <strong className="text-neon-cyan"> projected gradient descent</strong>. Starting from the
          pseudo-inverse solution as initial guess, we iteratively refine:
        </p>

        <div className="bg-dark-800 p-4 rounded border border-neon-green/30 mb-6">
          <div className="font-mono text-sm space-y-2">
            <p className="text-metal-silver">For each iteration:</p>
            <p className="text-neon-cyan pl-4">g = AᵀA·x - Aᵀb <span className="text-dark-300 ml-4">// gradient</span></p>
            <p className="text-neon-pink pl-4">α = (gᵀg) / (gᵀ·AᵀA·g) <span className="text-dark-300 ml-4">// optimal step size</span></p>
            <p className="text-neon-green pl-4">x = clip(x - α·g, 0, 1) <span className="text-dark-300 ml-4">// projected step</span></p>
          </div>
        </div>

        <p className="text-metal-silver mb-4">
          The <span className="font-mono text-neon-green">clip()</span> operation projects the solution
          back into the feasible region after each gradient step. The optimal step size{' '}
          <span className="font-mono text-neon-pink">α</span> ensures we make maximum progress toward
          the solution while maintaining stability.
        </p>

        <p className="text-metal-silver mb-4">
          In practice, <strong className="text-neon-cyan">5-15 iterations</strong> are sufficient to
          converge from the pseudo-inverse initialization, since it's already close to optimal.
        </p>
        <p className="text-metal-silver">
          The complex part of the above is the matrix multiplications with <span className="font-mono text-neon-cyan">AᵀA·x</span>,
          and <span className="font-mono text-neon-cyan">AᵀA·g</span>. To speed this up, we note
          that by suitably ordering the LEDs we can ensure that <span className="font-mono text-neon-cyan">AᵀA</span> is concentrated close to the diagonal.
          This can be done by ordering the LEDs so that LEDs that are far apart in the physical layout are also far apart in the ordering we use for
          computation. Since the LEDs are randomly arranged, we need to work out this ordering: we use Reverse Cuthill-McKee (RCM) algorithm to achieve this.
          The ordering can obviously be pre-computed, allowing us to store the precomputed <span className="font-mono text-neon-cyan">AᵀA</span>
          in this order, using a format efficient for a matrix with few non-zero diagonals. We provide custom GPU kernals for multiplying with this format.
        </p>
      </section>

      {/* Calibration */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Camera size={20} />
          Calibration: Learning the Display
        </h2>
        <p className="text-metal-silver mb-4">
          Before Prismatron can form images, it needs to learn its own diffusion patterns.
          The calibration process captures each LED's unique "fingerprint":
        </p>

        <ol className="space-y-3 text-metal-silver">
          <li className="flex items-start gap-3">
            <span className="text-neon-cyan font-retro">1.</span>
            <span><strong className="text-neon-cyan">Dark room setup</strong> — Camera positioned to capture the entire display</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="text-neon-cyan font-retro">2.</span>
            <span><strong className="text-neon-cyan">Sequential activation</strong> — Each LED lights up individually while a photo is captured</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="text-neon-cyan font-retro">3.</span>
            <span><strong className="text-neon-cyan">Pattern extraction</strong> — Software isolates each LED's diffusion pattern (64×64 crop)</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="text-neon-cyan font-retro">4.</span>
            <span><strong className="text-neon-cyan">Matrix precomputation</strong> — Build <span className="font-mono">AᵀA</span> and <span className="font-mono">(AᵀA)⁻¹</span></span>
          </li>
        </ol>

        <p className="text-metal-silver mt-6">
          Each pattern is stored as a 64×64 pixel crop around the LED's center, creating a library
          of 3,200 unique diffusion patterns—some are tight spots, others sprawling blooms,
          depending on the LED's position and the texture above it.
        </p>
      </section>

      {/* GPU Optimization */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Cpu size={20} />
          GPU Optimization: Tensor Cores
        </h2>
        <p className="text-metal-silver mb-4">
          The Jetson Orin Nano has <strong className="text-neon-green">Tensor Cores</strong>—specialized
          hardware for matrix operations. Accessing them requires careful data layout.
        </p>

        <div className="space-y-6">
          {/* Sparse Block Tensor for A^T */}
          <div className="border-l-2 border-neon-cyan pl-4">
            <h3 className="font-retro text-neon-cyan mb-2">Sparse Block Tensor for Aᵀ</h3>
            <p className="text-metal-silver text-sm mb-3">
              Each LED's diffusion pattern only affects a small 64×64 region of the image.
              Instead of storing the full sparse matrix, we use a <strong>single-block sparse tensor</strong>:
            </p>
            <div className="bg-dark-800 p-3 rounded font-mono text-xs text-neon-cyan">
              <p>sparse_values: (channels, LEDs, 64, 64) — dense blocks only</p>
              <p>block_positions: (channels, LEDs, 2) — top-left coordinates</p>
            </div>
            <p className="text-metal-silver text-sm mt-2">
              For <span className="font-mono">Aᵀb</span>, each LED extracts its 64×64 window from the
              target image and computes a dot product with its pattern—massively parallel on GPU.
            </p>
          </div>

          {/* Symmetric Block Diagonal for A^TA */}
          <div className="border-l-2 border-neon-pink pl-4">
            <h3 className="font-retro text-neon-pink mb-2">Symmetric Block Diagonal for AᵀA</h3>
            <p className="text-metal-silver text-sm mb-3">
              The Gram matrix <span className="font-mono">AᵀA</span> is symmetric and banded—LEDs only
              interact with nearby LEDs whose patterns overlap. We store it in{' '}
              <strong>symmetric block diagonal format</strong>:
            </p>
            <div className="bg-dark-800 p-3 rounded font-mono text-xs text-neon-pink">
              <p>block_data: (channels, block_diags, 16, 16) — 16×16 blocks</p>
              <p>block_offsets: (block_diags,) — diagonal positions</p>
            </div>
            <p className="text-metal-silver text-sm mt-2">
              Only upper diagonals are stored (symmetry gives us lower diagonals for free).
              Rather than storing each diagonal in an array, we decompose the whole matrix into 16×16 blocks
              and store the diagonals of blocks. This is a little less efficient (some blocks overlap empty
              diagonals) but it allows us to leverage Tensor Cores for acceleration.
            </p>
          </div>

          {/* WMMA Kernel */}
          <div className="border-l-2 border-neon-green pl-4">
            <h3 className="font-retro text-neon-green mb-2">Batching and custom WMMA Multiplication Kernel</h3>
            <p className="text-metal-silver text-sm mb-3">
              The algorithm described above involves matrix-vector multiplications, but Tensor Cores are designed
              for the matrix-matrix operations that power AI models. As a result, optimizing a single frame as described above
              happens only on the CUDA cores and we get only about 2 of the Jetson's 67 TFlops! The solution is to optimize frames 8 at a time.
              The required operations then become (8,32)x(32,8) matrix matrix multiplications which can be accelerated on the Tensor Cores.
              We wrote a <strong>custom CUDA kernel using WMMA intrinsics</strong> (Warp Matrix Multiply Accumulate):
            </p>
            <ul className="text-metal-silver text-sm space-y-1">
              <li className="flex items-start gap-2">
                <span className="text-neon-green">•</span>
                <span>Loads 16×16 matrix fragments into Tensor Core registers</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-neon-green">•</span>
                <span>Exploits symmetry: computes both upper and lower contributions</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-neon-green">•</span>
                <span>Processes 8 frames in parallel (batch mode)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-neon-green">•</span>
                <span>FP32 accumulation for numerical stability</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-6 p-4 bg-dark-700 rounded border-l-4 border-neon-purple">
          <h4 className="font-retro text-neon-purple text-sm mb-2">Why Tensor Cores?</h4>
          <p className="text-metal-silver text-sm">
            The Orin Nano's CUDA cores alone can't achieve real-time performance for 3,200 LEDs.
            Tensor Cores provide <strong className="text-neon-purple">8-16× speedup</strong> for
            matrix operations, but only with proper 16-byte alignment and specific data layouts.
            Our custom formats and kernels unlock this performance.
          </p>
        </div>
      </section>

      {/* Real-Time Performance */}
      <section className="content-card">
        <h2 className="subsection-title flex items-center gap-2">
          <Zap size={20} />
          Real-Time Performance
        </h2>
        <p className="text-metal-silver mb-4">
          The optimization pipeline processes each frame through:
        </p>

        <div className="bg-dark-800 rounded border border-neon-cyan/20 overflow-hidden mb-6">
          <table className="spec-table text-sm">
            <thead>
              <tr>
                <th>Stage</th>
                <th>Operation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="text-neon-cyan">1. Aᵀb</td>
                <td>Sparse block tensor dot product (custom kernel)</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">2. Initialize</td>
                <td>(AᵀA)⁻¹ · (Aᵀb) — pseudo-inverse starting point</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">3. Iterate</td>
                <td>5-15 gradient descent steps with WMMA kernel</td>
              </tr>
              <tr>
                <td className="text-neon-cyan">4. Output</td>
                <td>Clip and scale to [0, 255] for LED driver</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p className="text-metal-silver mb-4">
          In <strong className="text-neon-pink">batch mode</strong>, we process 8 frames simultaneously,
          maximizing GPU utilization. This does add delay into the optimization pipeline and we anyway maintain
          a faily deep buffer of optimized frames to smooth out any hiccups in performance. An optimized frame is
          just the RGB values for the 3200 LEDs and is relatively small compared to the original image frames.
        </p>
        <p className="text-metal-silver">
          Effects that need to be applied in real time (the audio-reactive effects) are applied to the LED values
          at rendering time, in order to make these as responsive as possible to the audio signal. These effects include
          fades, brightness modulation a random sparkle effect and various 'templates' which contain short pre-optimixed
          image sequences that can be blended in with the optimized output.
        </p>
      </section>

      <Comments pageSlug="algorithm" />
    </motion.div>
  )
}
