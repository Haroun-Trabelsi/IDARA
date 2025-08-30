import dotenv from 'dotenv'
dotenv.config()

import app from './utils/app' // (server)
import mongo from './utils/mongo' // (database)
import { PORT } from './constants/index'
import authRoutes from './routes/auth'
import projectsRouter from './routes/projects'

const bootstrap = async () => {
  await mongo.connect()

  app.get('/', (req, res) => {
    res.status(200).send('Hello, world!')
  })

  app.get('/healthz', (req, res) => {
    res.status(204).end()
  })

  app.use('/auth', authRoutes)
  app.use('/api', projectsRouter);

  app.listen(PORT, () => {
    console.log(`✅ Server is listening on port: ${PORT}`)
  }).on('error', (err) => {
    process.exit(1);
  });
}

bootstrap()
