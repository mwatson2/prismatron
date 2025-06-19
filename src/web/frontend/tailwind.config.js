/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // Retro-futurism color palette inspired by 1950s neon signs
      colors: {
        // Primary neon colors
        neon: {
          cyan: '#00ffff',
          pink: '#ff1493',
          green: '#39ff14',
          orange: '#ff6600',
          purple: '#bf00ff',
          yellow: '#ffff00',
          blue: '#0080ff'
        },
        // Dark backgrounds for contrast
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
        // Metallic accents
        metal: {
          silver: '#c0c0c0',
          chrome: '#e5e5e5',
          copper: '#b87333',
          brass: '#b5a642'
        }
      },

      // Typography for retro-futurism
      fontFamily: {
        'retro': ['Orbitron', 'monospace'],
        'display': ['Exo 2', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace']
      },

      // Box shadows for neon glow effects
      boxShadow: {
        'neon-sm': '0 0 5px currentColor',
        'neon': '0 0 10px currentColor, 0 0 20px currentColor',
        'neon-lg': '0 0 20px currentColor, 0 0 40px currentColor, 0 0 60px currentColor',
        'inner-neon': 'inset 0 0 10px currentColor'
      },

      // Animation for retro effects
      animation: {
        'pulse-neon': 'pulse-neon 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'flicker': 'flicker 0.15s infinite linear',
        'scan-line': 'scan-line 2s linear infinite',
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
        'scan-line': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100vw)' }
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

      // Grid patterns for retro layouts
      backgroundImage: {
        'grid-pattern': 'linear-gradient(rgba(0,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,255,0.1) 1px, transparent 1px)',
        'diagonal-stripes': 'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(0,255,255,0.1) 2px, rgba(0,255,255,0.1) 4px)',
        'neon-gradient': 'linear-gradient(45deg, #00ffff, #ff1493, #39ff14, #ff6600)'
      },

      backgroundSize: {
        'grid': '20px 20px',
        'stripes': '20px 20px'
      },

      // Spacing for consistent layouts
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem'
      },

      // Retro border radius
      borderRadius: {
        'retro': '0.125rem',
        'retro-lg': '0.25rem'
      }
    },
  },
  plugins: [
    // Custom plugin for neon text utility
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
        '.bg-neon': {
          'box-shadow': 'inset 0 0 10px currentColor',
        },
        '.retro-panel': {
          'background': 'linear-gradient(145deg, #1a1a1a, #0a0a0a)',
          'border': '1px solid rgba(0, 255, 255, 0.3)',
          'box-shadow': '0 0 10px rgba(0, 255, 255, 0.1), inset 0 0 10px rgba(0, 255, 255, 0.05)',
        },
        '.retro-button': {
          'background': 'linear-gradient(145deg, #262626, #121212)',
          'border': '1px solid currentColor',
          'box-shadow': '0 0 5px currentColor, inset 0 0 5px rgba(255, 255, 255, 0.1)',
          'transition': 'all 0.2s ease',
        },
        '.retro-button:hover': {
          'box-shadow': '0 0 15px currentColor, inset 0 0 5px rgba(255, 255, 255, 0.2)',
          'transform': 'translateY(-1px)',
        },
        '.retro-button:active': {
          'transform': 'translateY(0)',
          'box-shadow': '0 0 5px currentColor, inset 0 0 10px rgba(0, 0, 0, 0.3)',
        }
      }
      addUtilities(newUtilities)
    }
  ],
}
