import React from 'react'
import { MapPin, Phone, Mail } from 'lucide-react'
import { Link } from '@mui/material'

const About = () => {
  return (
    <footer className="bg-gray-900 text-gray-300 w-full">
      <div className="max-w-7xl mx-auto px-4 py-12 md:py-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Tunisia Office */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">Tunisian Location</h3>
            <div className="flex items-start space-x-2">
              <MapPin className="w-5 h-5 mt-0.5 flex-shrink-0" />
              <span>
                Immeuble Gahbiche, 2ème étage
                <br />
                Av. la Perle du Sahel Khezama,
                <br />
                Sousse 4051, Tunisie
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <Phone className="w-5 h-5 flex-shrink-0" />
              <Link href="tel:0021673205068" className="hover:text-white transition-colors">
                00 216 73 205 068
              </Link>
            </div>
            <div className="flex items-center space-x-2">
              <Mail className="w-5 h-5 flex-shrink-0" />
              <Link href="mailto:office@visionage-vfx.com" className="hover:text-white transition-colors">
                office@visionage-vfx.com
              </Link>
            </div>
          </div>

          {/* France Office */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">France</h3>
            <div className="flex items-center space-x-2">
              <Phone className="w-5 h-5 flex-shrink-0" />
              <Link href="tel:0033614231085" className="hover:text-white transition-colors">
                00 33 6 14 23 10 85
              </Link>
            </div>
            <div className="flex items-center space-x-2">
              <Mail className="w-5 h-5 flex-shrink-0" />
              <Link href="mailto:eu@visionage-vfx.com" className="hover:text-white transition-colors">
                eu@visionage-vfx.com
              </Link>
            </div>
          </div>

          {/* South Korea Office */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">South Korea</h3>
            <div className="space-y-1">
              <p>Hyejeong | VFX Coordinator</p>
            </div>
            <div className="flex items-center space-x-2">
              <Mail className="w-5 h-5 flex-shrink-0" />
              <Link href="mailto:korea@visionage-vfx.com" className="hover:text-white transition-colors">
                korea@visionage-vfx.com
              </Link>
            </div>
          </div>
        </div>

        {/* Headquarters & Copyright */}
        <div className="mt-12 pt-8 border-t border-gray-800 flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="text-sm">
              <span className="font-semibold text-white">Headquarter</span>
              <span className="mx-2">•</span>
              <span>Asma E.K | VFX Coordinator</span>
              <span className="mx-2">•</span>
              <Link href="mailto:asma@visionage-vfx.com" className="hover:text-white transition-colors">
                asma@visionage-vfx.com
              </Link>
            </p>
          </div>
          <div className="text-sm">
            <p>&copy; {new Date().getFullYear()} Visionage VFX. All rights reserved.</p>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default About
