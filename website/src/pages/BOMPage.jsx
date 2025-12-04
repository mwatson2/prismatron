import { motion } from 'framer-motion'
import { ShoppingCart, Box, Zap, Cpu, Wrench } from 'lucide-react'

const bomData = {
  Mechanical: [
    { name: 'Aluminum panels (back panel and electronics panel', link: 'https://www.nealscnc.com/' },
    { name: 'Acrylic diffusion panel and electronics panel cover', link: 'https://www.tapplastics.com/' },
    { name: 'Standoffs 1" x 4"', link: 'https://barrelsandcaps.com/products/1-diameter-4-tall-aluminum-barrell-1' },
    { name: 'Standoffs 0.75" x 2"', link: 'https://a.co/d/5RAyjSV' },
    { name: 'Washers (nylon)', link: 'https://a.co/d/0IbzpLg' },
    { name: 'Washers (Steel)', link: 'https://a.co/d/0YZdk5p' },
    { name: 'TV Stand', link: 'https://a.co/d/jg2lSnF' },
    { name: 'TV cover', link: 'https://a.co/d/iMAhcPH' },
    { name: 'Spacers', link: 'https://a.co/d/7p7dZak' },
    { name: 'Machine screws', link: 'https://a.co/d/gWqhjag' },
    { name: 'Cable clips', link: 'https://a.co/d/9hBGwab' },
    { name: 'Mini clips', link: 'https://a.co/d/d19Pbs2' },
    { name: 'Zip tie mounts', link: 'https://a.co/d/euiML6F' },
    { name: 'Small zip-tie mounts', link: 'https://a.co/d/iOwvIae' },
    { name: 'Zip ties', link: 'https://a.co/d/8jGXbvS' },
    { name: 'Enclosure for Jetson Orin Nano', link: 'https://a.co/d/7WRSCBd' },
    { name: 'Outer enclosure for Jetson Orin Nano', link: 'https://a.co/d/dZUqNCs' },
    { name: '5/16-18 Locknuts', link: 'https://a.co/d/2IK5y8I' },
    { name: '80 Mesh steel screen (dist protection)', link: 'https://a.co/d/ddYagnR' },
    { name: 'Velcro fasteners (for steel screen)', link: 'https://a.co/d/1Azgr7e' },
    { name: 'Fan cover/dust filters', link: 'https://a.co/d/1vN9jQs' },
    { name: 'VHB mounting tape', link: 'https://a.co/d/iUfKUjI' },
    { name: 'Metric locking nuts', link: 'https://a.co/d/in0kVgB' },

  ],
  Electrical: [
    { name: 'QuinLED Powerboard 7', link: 'https://www.drzzs.com/shop/octa-power-7/' },
    { name: 'MeanWell Power Supply 600W', link: 'https://a.co/d/gNErSe6' },
    { name: 'LED Strips (2.7mm addressable)', link: 'https://www.ledbe.com/2.7mm-individually-addressable-2812-led-strip' },
    { name: 'LED String lights (6x)', link: 'https://a.co/d/a41qhu6' },
    { name: '24AWG wire', link: 'https://a.co/d/eVWSNoe' },
    { name: '20AWG wire', link: 'https://a.co/d/ciXsVSC' },
    { name: '18AWG wire', link: 'https://a.co/d/27WkDcc' },
    { name: '16AWG Wire', link: 'https://a.co/d/5q3Qcfe' },
    { name: '14AWG Wire', link: 'https://a.co/d/6V61P9t' },
    { name: 'Power inlet', link: 'https://a.co/d/f0dzc8d' },
    { name: 'Spade connectors', link: 'https://a.co/d/4dBBnXm' },
    { name: 'Power cord (IEC320 C13)', link: 'https://a.co/d/6zmmtq9' },
    { name: 'Power cord (IEC320 C5)', link: 'https://a.co/d/4MryG0O' },
    { name: 'Etherenet socket/gland', link: 'https://a.co/d/hwH5KdN' },
    { name: 'Right angle audio jack', link: 'https://a.co/d/ivUfYTC' },
    { name: 'Audio panel socket', link: 'https://a.co/d/eutVPHg' },
    { name: 'USB Microphone', link: 'https://a.co/d/1SzenHS' },
    { name: 'Heat shrink butt connectors', link: 'https://a.co/d/4TVjAck' },
    { name: 'Cooling fan', link: 'https://a.co/d/d5GkJud' },
    { name: 'Wifi Antenna extenders', link: 'https://a.co/d/9sd8FVd' },
    { name: 'Inline power meter', link: 'https://a.co/d/hrYXsW5' },
    { name: 'Spade connectors', link: 'https://a.co/d/afG1Y8B' },
  ],
  Compute: [
    { name: 'QuinLED Digi-Octa Brainboard', link: 'https://www.drzzs.com/shop/dig-octa/' },
    { name: 'Jetson Orin Nano Super', link: 'https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit' },
    { name: 'SSD for Jetson', link: 'https://a.co/d/7QpvMI4' },
    { name: 'Camera (for calibration)', link: 'https://a.co/d/4YUXPLG' },
  ],
  Tools: [
    { name: 'Ferrule crimp kit', link: 'https://a.co/d/hGIRW7R' },
    { name: 'JST-SM Tool and connectors', link: 'https://a.co/d/fZmHbbg' },
    { name: 'Workbench', link: 'https://a.co/d/gYpgDa1' },
    { name: 'Heat gun', link: 'https://a.co/d/abdQjT9' },
    { name: 'Drill bits', link: 'https://a.co/d/2qjm4S0' },
    { name: 'Step Drill bits', link: 'https://a.co/d/e4YaAMe' },
    { name: 'Center punch', link: 'https://a.co/d/c5idDtX' },
    { name: 'Ruler', link: 'https://a.co/d/265ojx4' },
    { name: 'Layout fluid', link: 'https://a.co/d/diffDmL' },
    { name: 'Scribe tool', link: 'https://a.co/d/dR9FPrn' },
    { name: 'Loctite blue 242', link: 'https://a.co/d/gGt6YBE' },
    { name: 'Gorilla Epoxy', link: 'https://a.co/d/da3XZoO' },
    { name: 'Hole saw', link: 'https://a.co/d/7b2F8cr' },
    { name: 'Deburring kit', link: 'https://a.co/d/5H1LusV' },
    { name: 'Small cutting tool', link: 'https://a.co/d/hivxv6V' },
    { name: 'Crimp tool', link: 'https://a.co/d/35lwzVv' },
    { name: 'Self-adjusting wire stripper', link: 'https://a.co/d/gQ64Cxs' },
  ],
}

const categoryConfig = {
  Mechanical: { icon: Box, color: 'neon-cyan' },
  Electrical: { icon: Zap, color: 'neon-yellow' },
  Compute: { icon: Cpu, color: 'neon-green' },
  Tools: { icon: Wrench, color: 'neon-purple' },
}

export default function BOMPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <h1 className="section-title flex items-center gap-3">
        <ShoppingCart className="text-neon-orange" />
        Bill of Materials
      </h1>

      <section className="content-card mb-6">
        <p className="text-metal-silver">
          Here's what went into building Prismatron. Links are provided for reference—prices
          and availability may vary. Some items were sourced from local suppliers or eBay.
        </p>
      </section>

      {Object.entries(bomData).map(([category, items]) => {
        const config = categoryConfig[category]
        const Icon = config.icon

        return (
          <section key={category} className="content-card">
            <h2 className={`subsection-title flex items-center gap-2 text-${config.color}`}>
              <Icon size={20} />
              {category}
            </h2>

            <ul className="space-y-2">
              {items.map((item, index) => (
                <li key={index} className="flex items-center gap-2">
                  <span className={`text-${config.color}`}>•</span>
                  {item.link ? (
                    <a
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-metal-silver hover:text-neon-cyan transition-colors"
                    >
                      {item.name}
                    </a>
                  ) : (
                    <span className="text-metal-silver">{item.name}</span>
                  )}
                </li>
              ))}
            </ul>
          </section>
        )
      })}
    </motion.div>
  )
}
