import mongoose from 'mongoose'
import { MONGO_URI, MONGO_OPTIONS } from '../constants/index'
import dotenv from 'dotenv';

dotenv.config();
class Mongo {
  instance: typeof mongoose = mongoose
  mongoUri: string
  mongoOptions: mongoose.ConnectOptions
  isConnected: boolean

  constructor() {
    this.mongoUri = process.env.MONGO_URI ||'env not working'
    this.mongoOptions = MONGO_OPTIONS
    this.isConnected = false
  }

async connect() {
  if (this.isConnected) return

  try {
    console.log('⏳ Connecting to MongoDB')

    const options = {
      ...this.mongoOptions,
      dbName: 'vfx_classification_demo',  // explicitly specify database here
    }

    const db = await this.instance.connect(this.mongoUri, options)
    const connection = db.connection

    this.isConnected = connection.readyState === 1
    if (this.isConnected) console.log('✅ MongoDB connected')

    connection.on('connected', () => console.log('✅ MongoDB connected')) // re-connected
    connection.on('disconnected', () => console.log('❌ MongoDB disconnected')) // disconnected
    connection.on('error', (error) => console.log('❌ MongoDB connection error', error)) // listen for errors during the session
  } catch (error: any) {
    console.log('❌ MongoDB connection error:', error.message)
  }
}

}

export default new Mongo()
