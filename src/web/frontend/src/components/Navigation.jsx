import React from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
  HomeIcon,
  CloudArrowUpIcon,
  FolderIcon,
  SparklesIcon,
  QueueListIcon,
  CogIcon
} from '@heroicons/react/24/outline'
import {
  HomeIcon as HomeSolidIcon,
  CloudArrowUpIcon as CloudArrowUpSolidIcon,
  FolderIcon as FolderSolidIcon,
  SparklesIcon as SparklesSolidIcon,
  QueueListIcon as QueueListSolidIcon,
  CogIcon as CogSolidIcon
} from '@heroicons/react/24/solid'

const Navigation = () => {
  const location = useLocation()

  const navItems = [
    {
      path: '/home',
      label: 'HOME',
      Icon: HomeIcon,
      IconSolid: HomeSolidIcon,
      description: 'Preview & Control'
    },
    {
      path: '/upload',
      label: 'UPLOAD',
      Icon: CloudArrowUpIcon,
      IconSolid: CloudArrowUpSolidIcon,
      description: 'Add Content'
    },
    {
      path: '/media',
      label: 'MEDIA',
      Icon: FolderIcon,
      IconSolid: FolderSolidIcon,
      description: 'Manage Files'
    },
    {
      path: '/effects',
      label: 'EFFECTS',
      Icon: SparklesIcon,
      IconSolid: SparklesSolidIcon,
      description: 'Visual Effects'
    },
    {
      path: '/playlist',
      label: 'PLAYLIST',
      Icon: QueueListIcon,
      IconSolid: QueueListSolidIcon,
      description: 'Manage Queue'
    },
    {
      path: '/settings',
      label: 'SETTINGS',
      Icon: CogIcon,
      IconSolid: CogSolidIcon,
      description: 'System Config'
    }
  ]

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-40">
      {/* Retro panel background */}
      <div className="retro-panel mx-2 mb-2 rounded-retro-lg border-t border-neon-cyan border-opacity-30 backdrop-blur-sm">
        <div className="flex justify-between items-center py-2 px-1">
          {navItems.map(({ path, label, Icon, IconSolid, description }) => {
            const isActive = location.pathname === path
            const IconComponent = isActive ? IconSolid : Icon

            return (
              <NavLink
                key={path}
                to={path}
                className={`nav-item group relative flex-1 max-w-[60px] ${
                  isActive ? 'nav-item-active' : 'nav-item-inactive'
                }`}
                aria-label={`${label}: ${description}`}
              >
                {/* Icon */}
                <div className="relative mb-1">
                  <IconComponent
                    className={`w-5 h-5 sm:w-6 sm:h-6 transition-all duration-200 ${
                      isActive
                        ? 'drop-shadow-[0_0_8px_currentColor]'
                        : 'group-hover:drop-shadow-[0_0_4px_currentColor]'
                    }`}
                  />

                  {/* Active indicator glow */}
                  {isActive && (
                    <div className="absolute inset-0 w-5 h-5 sm:w-6 sm:h-6 rounded-full bg-neon-cyan opacity-20 blur-sm animate-pulse-neon" />
                  )}
                </div>

                {/* Label - hide on very small screens if needed */}
                <span className={`text-[9px] sm:text-[10px] font-bold tracking-wider ${
                  isActive ? 'text-shadow-neon' : ''
                }`}>
                  {label}
                </span>

                {/* Hover tooltip - only show on larger screens */}
                <div className="hidden sm:block absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-dark-800 text-neon-cyan text-xs rounded-retro opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap border border-neon-cyan border-opacity-30">
                  {description}
                  <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-2 border-r-2 border-t-2 border-transparent border-t-neon-cyan border-opacity-30" />
                </div>
              </NavLink>
            )
          })}
        </div>

        {/* Bottom accent line */}
        <div className="h-0.5 bg-gradient-to-r from-transparent via-neon-cyan to-transparent opacity-30" />
      </div>
    </nav>
  )
}

export default Navigation
