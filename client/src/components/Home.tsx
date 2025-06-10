
import React from 'react'
import { Container, Typography, Box } from '@mui/material'

const Home = () => {
  return (
    <Container maxWidth="lg" >
      <Box sx={{ my: 4 }}>
        <Typography variant="h2" component="h1" gutterBottom>
          Welcome to Our App
        </Typography>
        <Typography variant="h5" component="h2" gutterBottom>
          Get started by exploring our features
        </Typography>
      </Box>
    </Container>
  )
}

export default Home
