/**
 * Database Seeding Script
 * Run this script to populate the database with dummy data for demonstration
 * 
 * Usage: npm run seed
 */

import mongoose from 'mongoose';
import seedDatabase from '../utils/seedData';

// Load environment variables
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/idara';

async function main() {
  try {
    console.log('🔌 Connecting to MongoDB...');
    console.log(`   URI: ${MONGODB_URI.replace(/\/\/([^:]+):([^@]+)@/, '//***:***@')}`);
    
    await mongoose.connect(MONGODB_URI);
    console.log('✅ Connected to MongoDB');

    await seedDatabase();

    console.log('\n✨ All done! You can now use the application with demo data.');
    process.exit(0);
  } catch (error) {
    console.error('💥 Fatal error:', error);
    process.exit(1);
  }
}

main();

