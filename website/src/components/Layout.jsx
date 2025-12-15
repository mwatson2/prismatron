import { useState, useEffect } from 'react'
import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { Menu, X, Cpu, Zap, Box, Code, Music, Settings, Home, FileText, Smartphone, Scale, ShoppingCart, Flame } from 'lucide-react'
import clsx from 'clsx'
import Footer from './Footer'

const navItems = [
  { path: '/', label: 'Home', icon: Home },
  { path: '/burning-man', label: 'Burning Man', icon: Flame },
  { path: '/mechanical', label: 'Mechanical', icon: Box },
  { path: '/electrical', label: 'Electrical', icon: Zap },
  { path: '/compute', label: 'Compute', icon: Cpu },
  { path: '/algorithm', label: 'Algorithm', icon: Code },
  { path: '/audio', label: 'Audio-Reactive', icon: Music },
  { path: '/software', label: 'Software', icon: Settings },
  { path: '/control-app', label: 'Control App', icon: Smartphone },
  { path: '/specs', label: 'Specifications', icon: FileText },
  { path: '/bom', label: 'Bill of Materials', icon: ShoppingCart },
  { path: '/license', label: 'License', icon: Scale },
]

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  // Scroll to top on route change
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [location.pathname])

  return (
    <div className="min-h-screen grid-bg">
      {/* Mobile header */}
      <header className="lg:hidden fixed top-0 left-0 right-0 z-50 retro-panel border-b border-neon-cyan/30">
        <div className="flex items-center justify-between p-4">
          <NavLink to="/" className="font-retro text-xl text-neon-cyan text-neon">
            PRISMATRON
          </NavLink>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 text-neon-cyan hover:text-neon-pink transition-colors"
          >
            {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </header>

      {/* Sidebar */}
      <aside
        className={clsx(
          'fixed top-0 left-0 h-full w-64 retro-panel border-r border-neon-cyan/30 z-40 transition-transform duration-300',
          'lg:translate-x-0',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="p-6 border-b border-neon-cyan/30">
          <NavLink to="/" className="block" onClick={() => setSidebarOpen(false)}>
            <h1 className="font-retro text-2xl text-neon-cyan text-neon tracking-wider">
              PRISMATRON
            </h1>
            <p className="text-xs text-metal-silver mt-1">
              Computational LED Display
            </p>
          </NavLink>
        </div>

        <nav className="py-4">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              onClick={() => setSidebarOpen(false)}
              className={({ isActive }) =>
                clsx('nav-link flex items-center gap-3', isActive && 'nav-link-active')
              }
            >
              <item.icon size={18} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-neon-cyan/30">
          <p className="text-xs text-dark-300 text-center">
            Where chaos becomes coherent
          </p>
        </div>
      </aside>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <main className="lg:ml-64 min-h-screen pt-16 lg:pt-0">
        <div className="max-w-4xl mx-auto p-6 lg:p-10">
          <Outlet />
          <Footer />
        </div>
      </main>
    </div>
  )
}
