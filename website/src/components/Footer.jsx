export default function Footer() {
  return (
    <footer className="mt-12 pt-6 border-t border-dark-600">
      <p className="text-dark-300 text-xs text-center leading-relaxed">
        Licensed under{' '}
        <a
          href="https://polyformproject.org/licenses/noncommercial/1.0.0/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-neon-cyan/70 hover:text-neon-cyan transition-colors"
        >
          Polyform NC 1.0.0
        </a>{' '}
        (code) and{' '}
        <a
          href="https://creativecommons.org/licenses/by-nc/4.0/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-neon-cyan/70 hover:text-neon-cyan transition-colors"
        >
          CC BY-NC 4.0
        </a>{' '}
        (other materials).
        <br />
        Free for personal, research, and art projects with attribution.{' '}
        <a
          href="mailto:markwatson@cantab.net"
          className="text-neon-pink/70 hover:text-neon-pink transition-colors"
        >
          Contact me
        </a>{' '}
        for commercial licensing.
      </p>
    </footer>
  )
}
