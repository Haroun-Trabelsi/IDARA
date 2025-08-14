import React from 'react';
import { MapPin, Phone, Mail } from 'lucide-react';
import { Link, Box, Typography } from '@mui/material';

const About = () => {
  return (
    <Box component="footer" sx={{ backgroundColor: '#1f2937', color: '#d1d5db', width: '100%' }}>
      <Box sx={{ maxWidth: '1200px', margin: '0 auto', padding: { xs: '2rem 1rem', md: '4rem 1rem' } }}>
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: '2rem' }}>
          {/* Tunisia Office */}
          <Box>
            <Typography variant="h6" sx={{ color: 'white' }}>Tunisian Location</Typography>
            <Box sx={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
              <MapPin size={20} style={{ marginTop: '0.25rem' }} />
              <Typography variant="body2">
                Immeuble Gahbiche, 2ème étage<br />
                Av. la Perle du Sahel Khezama,<br />
                Sousse 4051, Tunisie
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
              <Phone size={20} />
              <Link href="tel:0021673205068" underline="hover" color="inherit">
                00 216 73 205 068
              </Link>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
              <Mail size={20} />
              <Link href="mailto:office@visionage-vfx.com" underline="hover" color="inherit">
                office@visionage-vfx.com
              </Link>
            </Box>
          </Box>

          {/* France Office */}
          <Box>
            <Typography variant="h6" sx={{ color: 'white' }}>France</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
              <Phone size={20} />
              <Link href="tel:0033614231085" underline="hover" color="inherit">
                00 33 6 14 23 10 85
              </Link>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
              <Mail size={20} />
              <Link href="mailto:eu@visionage-vfx.com" underline="hover" color="inherit">
                eu@visionage-vfx.com
              </Link>
            </Box>
          </Box>

          {/* South Korea Office */}
          <Box>
            <Typography variant="h6" sx={{ color: 'white' }}>South Korea</Typography>
            <Typography variant="body2" sx={{ marginTop: '0.5rem' }}>Hyejeong | VFX Coordinator</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
              <Mail size={20} />
              <Link href="mailto:korea@visionage-vfx.com" underline="hover" color="inherit">
                korea@visionage-vfx.com
              </Link>
            </Box>
          </Box>
        </Box>

        {/* HQ & Copyright */}
        <Box sx={{
          marginTop: '3rem',
          paddingTop: '2rem',
          borderTop: '1px solid #374151',
          display: 'flex',
          flexDirection: { xs: 'column', md: 'row' },
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '1rem',
        }}>
          <Typography variant="body2" align="center">
            <strong style={{ color: 'white' }}>Headquarter</strong> • Asma E.K | VFX Coordinator •{' '}
            <Link href="mailto:asma@visionage-vfx.com" underline="hover" color="inherit">
              asma@visionage-vfx.com
            </Link>
          </Typography>
          <Typography variant="body2" align="center">
            &copy; {new Date().getFullYear()} Visionage VFX. All rights reserved.
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default About;