import dotenv from 'dotenv'
dotenv.config()

import app from './utils/app' // (server)
import mongo from './utils/mongo' // (database)
import { PORT } from './constants/index'
import authRoutes from './routes/auth'
import projectsRouter from './routes/projects'
import collaborator from './routes/collaborator'
import contactRoutes from './routes/contact';

import adminRoutes from './routes/admin';



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
  app.use('/col', collaborator);
  app.use('/contact', contactRoutes);
  app.use('/admin', adminRoutes);


  app.listen(PORT, () => {
    console.log(`✅ Server is listening on port: ${PORT}`)
  }).on('error', (err) => {
    process.exit(1);
  });
}

bootstrap()
