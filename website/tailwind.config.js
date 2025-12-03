/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        neon: {
          cyan: '#00ffff',
          pink: '#ff1493',
          green: '#39ff14',
          orange: '#ff6600',
          purple: '#bf00ff',
          yellow: '#ffff00',
          blue: '#0080ff'
        },
        dark: {
          900: '#0a0a0a',
          800: '#121212',
          700: '#1a1a1a',
          600: '#262626',
          500: '#333333',
          400: '#404040',
          300: '#595959',
          200: '#737373',
          100: '#8c8c8c'
        },
        metal: {
          silver: '#c0c0c0',
          chrome: '#e5e5e5',
          copper: '#b87333',
          brass: '#b5a642'
        }
      },
      fontFamily: {
        'retro': ['Orbitron', 'monospace'],
        'display': ['Exo 2', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace']
      },
      boxShadow: {
        'neon-sm': '0 0 5px currentColor',
        'neon': '0 0 10px currentColor, 0 0 20px currentColor',
        'neon-lg': '0 0 20px currentColor, 0 0 40px currentColor, 0 0 60px currentColor',
        'inner-neon': 'inset 0 0 10px currentColor'
      },
      animation: {
        'pulse-neon': 'pulse-neon 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'flicker': 'flicker 0.15s infinite linear',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 3s ease-in-out infinite'
      },
      keyframes: {
        'pulse-neon': {
          '0%, 100%': {
            opacity: '1',
            filter: 'brightness(1) drop-shadow(0 0 5px currentColor)'
          },
          '50%': {
            opacity: '0.8',
            filter: 'brightness(1.2) drop-shadow(0 0 15px currentColor)'
          }
        },
        'flicker': {
          '0%, 19%, 21%, 23%, 25%, 54%, 56%, 100%': {
            opacity: '1',
            filter: 'brightness(1)'
          },
          '20%, 24%, 55%': {
            opacity: '0.4',
            filter: 'brightness(0.4)'
          }
        },
        'glow': {
          'from': {
            filter: 'brightness(1) drop-shadow(0 0 5px currentColor)'
          },
          'to': {
            filter: 'brightness(1.3) drop-shadow(0 0 20px currentColor)'
          }
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' }
        }
      },
      backgroundImage: {
        'grid-pattern': 'linear-gradient(rgba(0,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,255,0.1) 1px, transparent 1px)',
      },
      backgroundSize: {
        'grid': '20px 20px',
      },
    },
  },
  plugins: [
    function({ addUtilities }) {
      const newUtilities = {
        '.text-neon': {
          'text-shadow': '0 0 5px currentColor, 0 0 10px currentColor, 0 0 15px currentColor',
        },
        '.text-neon-strong': {
          'text-shadow': '0 0 5px currentColor, 0 0 10px currentColor, 0 0 15px currentColor, 0 0 20px currentColor',
        },
        '.border-neon': {
          'box-shadow': '0 0 5px currentColor, inset 0 0 5px currentColor',
        },
        '.retro-panel': {
          'background': 'linear-gradient(145deg, #1a1a1a, #0a0a0a)',
          'border': '1px solid rgba(0, 255, 255, 0.3)',
          'box-shadow': '0 0 10px rgba(0, 255, 255, 0.1), inset 0 0 10px rgba(0, 255, 255, 0.05)',
        },
      }
      addUtilities(newUtilities)
    }
  ],
}
