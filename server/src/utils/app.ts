import express from 'express'
import cors from 'cors'
import { ORIGIN } from '../constants/index'

// initialize app
const app = express()


// middlewares
app.use(cors({
    origin: 'http://localhost:3000',
    credentials: true,
    allowedHeaders: ['Content-Type', 'Authorization'],
  }));
  
  app.use(express.json()) // body parser
app.use(express.urlencoded({ extended: false })) // url parser

export default app
