/**
 * Seed Data for IDara Backend
 * This file contains dummy data to populate the MongoDB database
 * for demonstration purposes when API access is unavailable.
 */

import Account from '../models/Account';
import ContactMessage from '../models/ContactMessage';
import Feedback from '../models/Feedback';
import Result from '../models/Result';

// Dummy password (hashed version of "Demo123!")
// In production, this would be properly hashed using bcrypt
const DEMO_PASSWORD_HASH = '$2a$10$dummyHashForDemonstrationPurposesOnly';

export const seedAccounts = [
  {
    name: "John",
    surname: "Doe",
    username: "johndoe",
    email: "john.doe@vfxstudio.com",
    password: DEMO_PASSWORD_HASH,
    role: "admin",
    isVerified: true,
    mfaEnabled: false,
    organizationName: "VFX Studio Alpha",
    status: "AdministratorOrganization",
    canInvite: true,
    mustCompleteProfile: false,
    teamSize: "10-50",
    region: "North America",
    ID_Organization: "ORG-001",
    companyFtrackLink: "https://vfxstudio.ftrackapp.com",
    apiKey: "demo_api_key_1234567890abcdef",
    lastConnexion: new Date("2025-10-15T10:30:00Z"),
    receiveUpdates: true,
  },
  {
    name: "Sarah",
    surname: "Johnson",
    username: "sarahj",
    email: "sarah.johnson@vfxstudio.com",
    password: DEMO_PASSWORD_HASH,
    role: "user",
    isVerified: true,
    mfaEnabled: true,
    organizationName: "VFX Studio Alpha",
    status: "accepted",
    canInvite: true,
    mustCompleteProfile: false,
    teamSize: "10-50",
    region: "North America",
    lastConnexion: new Date("2025-10-14T16:45:00Z"),
    receiveUpdates: true,
  },
  {
    name: "Mike",
    surname: "Chen",
    username: "mikechen",
    email: "mike.chen@creativevfx.com",
    password: DEMO_PASSWORD_HASH,
    role: "admin",
    isVerified: true,
    mfaEnabled: false,
    organizationName: "Creative VFX House",
    status: "AdministratorOrganization",
    canInvite: true,
    mustCompleteProfile: false,
    teamSize: "50-200",
    region: "Asia Pacific",
    ID_Organization: "ORG-002",
    companyFtrackLink: "https://creativevfx.ftrackapp.com",
    apiKey: "demo_api_key_0987654321fedcba",
    lastConnexion: new Date("2025-10-15T03:20:00Z"),
    receiveUpdates: false,
  },
  {
    name: "Emma",
    surname: "Williams",
    username: "emmaw",
    email: "emma.williams@vfxstudio.com",
    password: DEMO_PASSWORD_HASH,
    role: "user",
    isVerified: true,
    mfaEnabled: false,
    organizationName: "VFX Studio Alpha",
    status: "accepted",
    canInvite: false,
    mustCompleteProfile: false,
    teamSize: "10-50",
    region: "North America",
    lastConnexion: new Date("2025-10-13T14:20:00Z"),
    receiveUpdates: true,
  },
  {
    name: "Alex",
    surname: "Rodriguez",
    username: "alexr",
    email: "alex.rodriguez@vfxstudio.com",
    password: DEMO_PASSWORD_HASH,
    role: "user",
    isVerified: false,
    mfaEnabled: false,
    organizationName: "VFX Studio Alpha",
    status: "pending",
    canInvite: false,
    mustCompleteProfile: true,
    teamSize: "10-50",
    region: "North America",
    invitedDate: new Date("2025-10-10T09:00:00Z"),
    receiveUpdates: false,
  },
];

export const seedResults = [
  {
    filename: "GW_SQ010_SH0010_v001.mov",
    project: "Galactic Warriors",
    sequence: "SQ010",
    shot: "SH0010",
    classification_result: {
      predicted_class: "Hard",
      confidence: "0.87",
      probabilities: new Map([
        ["Easy", "0.05"],
        ["Medium", "0.08"],
        ["Hard", "0.87"],
      ]),
    },
    complexity_scores: {
      blur_score: 0.72,
      focus_pull_score: 0.85,
      noise_score: 0.45,
      camera_motion_score: 0.91,
      parallax_score: 0.78,
      overlap_score: 0.82,
      distortion_score: 0.56,
      lighting_score: 0.68,
      zoom_score: 0.73,
    },
    predicted_vfx_hours: 48.5,
    processing_time_seconds: 12.4,
    version: 1,
  },
  {
    filename: "GW_SQ010_SH0020_v001.mov",
    project: "Galactic Warriors",
    sequence: "SQ010",
    shot: "SH0020",
    classification_result: {
      predicted_class: "Medium",
      confidence: "0.76",
      probabilities: new Map([
        ["Easy", "0.12"],
        ["Medium", "0.76"],
        ["Hard", "0.12"],
      ]),
    },
    complexity_scores: {
      blur_score: 0.35,
      focus_pull_score: 0.42,
      noise_score: 0.28,
      camera_motion_score: 0.55,
      parallax_score: 0.48,
      overlap_score: 0.52,
      distortion_score: 0.31,
      lighting_score: 0.44,
      zoom_score: 0.38,
    },
    predicted_vfx_hours: 24.2,
    processing_time_seconds: 9.8,
    version: 1,
  },
  {
    filename: "GW_SQ020_SH0010_v002.mov",
    project: "Galactic Warriors",
    sequence: "SQ020",
    shot: "SH0010",
    classification_result: {
      predicted_class: "Hard",
      confidence: "0.92",
      probabilities: new Map([
        ["Easy", "0.02"],
        ["Medium", "0.06"],
        ["Hard", "0.92"],
      ]),
    },
    complexity_scores: {
      blur_score: 0.88,
      focus_pull_score: 0.91,
      noise_score: 0.67,
      camera_motion_score: 0.94,
      parallax_score: 0.89,
      overlap_score: 0.93,
      distortion_score: 0.72,
      lighting_score: 0.81,
      zoom_score: 0.86,
    },
    predicted_vfx_hours: 67.8,
    processing_time_seconds: 15.2,
    version: 2,
  },
  {
    filename: "OM_SQ010_SH0005_v001.mov",
    project: "Ocean's Mystery",
    sequence: "SQ010",
    shot: "SH0005",
    classification_result: {
      predicted_class: "Easy",
      confidence: "0.82",
      probabilities: new Map([
        ["Easy", "0.82"],
        ["Medium", "0.14"],
        ["Hard", "0.04"],
      ]),
    },
    complexity_scores: {
      blur_score: 0.15,
      focus_pull_score: 0.22,
      noise_score: 0.18,
      camera_motion_score: 0.25,
      parallax_score: 0.21,
      overlap_score: 0.28,
      distortion_score: 0.19,
      lighting_score: 0.32,
      zoom_score: 0.16,
    },
    predicted_vfx_hours: 8.5,
    processing_time_seconds: 7.3,
    version: 1,
  },
  {
    filename: "OM_SQ020_SH0012_v001.mov",
    project: "Ocean's Mystery",
    sequence: "SQ020",
    shot: "SH0012",
    classification_result: {
      predicted_class: "Medium",
      confidence: "0.71",
      probabilities: new Map([
        ["Easy", "0.18"],
        ["Medium", "0.71"],
        ["Hard", "0.11"],
      ]),
    },
    complexity_scores: {
      blur_score: 0.48,
      focus_pull_score: 0.53,
      noise_score: 0.41,
      camera_motion_score: 0.59,
      parallax_score: 0.51,
      overlap_score: 0.56,
      distortion_score: 0.44,
      lighting_score: 0.49,
      zoom_score: 0.47,
    },
    predicted_vfx_hours: 28.3,
    processing_time_seconds: 10.5,
    version: 1,
  },
];

export const seedContactMessages = [
  {
    name: "David Thompson",
    email: "david.thompson@email.com",
    subject: "Question about enterprise pricing",
    message: "Hi, I'm interested in learning more about your enterprise plans for larger VFX studios. Could you provide more information?",
  },
  {
    name: "Lisa Martinez",
    email: "lisa.m@filmproduction.com",
    subject: "Integration with ftrack",
    message: "Does your platform integrate with ftrack? We're using it for project management and would like to know about compatibility.",
  },
  {
    name: "Robert Lee",
    email: "robert.lee@studio.com",
    subject: "Bug report - upload feature",
    message: "I'm experiencing issues when uploading large video files (>2GB). The upload seems to timeout. Can you look into this?",
  },
];

export const seedFeedback = [
  {
    rating: 5,
    feedbackText: "The complexity analysis is incredibly accurate and has saved our team countless hours in estimation. Highly recommend!",
    featureSuggestions: ["Batch processing for multiple shots", "Export reports to PDF"],
  },
  {
    rating: 4,
    feedbackText: "Great tool overall. The UI is clean and intuitive. Would love to see more customization options for the dashboard.",
    featureSuggestions: ["Custom dashboard widgets", "Dark mode", "Mobile app"],
  },
  {
    rating: 5,
    feedbackText: "The VFX hour predictions are spot-on! This has revolutionized how we bid on projects.",
    featureSuggestions: ["API access for custom integrations", "Real-time collaboration features"],
  },
];

/**
 * Seed the database with dummy data
 */
export async function seedDatabase() {
  try {
    console.log('🌱 Starting database seeding...');

    // Clear existing data (optional - comment out if you want to keep existing data)
    // await Account.deleteMany({});
    // await Result.deleteMany({});
    // await ContactMessage.deleteMany({});
    // await Feedback.deleteMany({});

    // Seed Accounts
    console.log('📝 Seeding accounts...');
    const createdAccounts = await Account.insertMany(seedAccounts);
    console.log(`✅ Created ${createdAccounts.length} accounts`);

    // Seed Results
    console.log('📊 Seeding results...');
    const createdResults = await Result.insertMany(seedResults);
    console.log(`✅ Created ${createdResults.length} results`);

    // Seed Contact Messages
    console.log('✉️ Seeding contact messages...');
    const createdMessages = await ContactMessage.insertMany(seedContactMessages);
    console.log(`✅ Created ${createdMessages.length} contact messages`);

    // Seed Feedback (link to first account)
    console.log('💬 Seeding feedback...');
    const feedbackWithAccounts = seedFeedback.map((fb, index) => ({
      ...fb,
      accountId: createdAccounts[Math.min(index, createdAccounts.length - 1)]._id,
    }));
    const createdFeedback = await Feedback.insertMany(feedbackWithAccounts);
    console.log(`✅ Created ${createdFeedback.length} feedback entries`);

    console.log('🎉 Database seeding completed successfully!');
    console.log('\n📋 Demo Credentials:');
    console.log('   Email: john.doe@vfxstudio.com');
    console.log('   Password: Demo123!');
    
    return {
      accounts: createdAccounts,
      results: createdResults,
      messages: createdMessages,
      feedback: createdFeedback,
    };
  } catch (error) {
    console.error('❌ Error seeding database:', error);
    throw error;
  }
}

export default seedDatabase;

