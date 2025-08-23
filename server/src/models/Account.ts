import { type Document, model, Schema } from 'mongoose';
import { type Account } from '../@types';

interface I extends Document, Account {
  _id: string;
  createdAt: Date;
  updatedAt: Date;
  organizationName: string;
  invitedBy?: string;
  canInvite: boolean;
  status?: 'pending' | 'accepted' | 'expired' | 'AdministratorOrganization';
  invitedDate?: Date;
  verificationToken?: string;
  verificationTokenExpires?: Date;
  mustCompleteProfile: boolean;
  receiveUpdates: boolean;
  feedbackText?: string;
  featureSuggestions?: string[];
  rating?: number;
  teamSize: string;
  region: string;
  ID_Organization?: string;
  lastConnexion?: Date; // Nouveau champ pour la dernière connexion
  resetPasswordToken?:string ; // Nouveau champ pour le jeton de réinitialisation

}

const instance = new Schema<I>(
  {
    name: {
      type: String,
      required: false,
    },
    username: {
      type: String,
      required: false,
      trim: true,
    },
    companyFtrackLink: {
      type: String,
      required: false,
      trim: true,
    },
    apiKey: {
      type: String,
      required: false,
      select: false, // Do not return by default for security
    },
    surname: {
      type: String,
      required: false,
    },
    email: {
      type: String,
      required: true,
      lowercase: true,
      unique: true,
    },
    password: {
      type: String,
      required: true,
    },
    role: {
      type: String,
      required: true,
      enum: ['user', 'admin'],
      default: 'user',
    },
    isVerified: {
      type: Boolean,
      required: true,
      default: false,
    },
    verificationCode: {
      type: String,
      required: false,
    },
    verificationCodeExpires: {
      type: Date,
      required: false,
    },
    mfaSecret: { 
      type: String,
    },
    mfaEnabled: { 
      type: Boolean,
      default: false,
    },
    organizationName: {
      type: String,
      required: true,
    },
    invitedBy: {
      type: Schema.Types.ObjectId,
      ref: 'Account',
      required: false,
    },
    canInvite: {
      type: Boolean,
      default: true,
    },
    status: {
      type: String,
      enum: ['pending', 'accepted', 'expired', 'AdministratorOrganization'],
      default: 'AdministratorOrganization',
    },
    invitedDate: {
      type: Date,
      required: false,
    },
    verificationToken: {
      type: String,
      required: false,
    },
    verificationTokenExpires: {
      type: Date,
      required: false,
    },
    mustCompleteProfile: { 
      type: Boolean, 
      default: false 
    },
    receiveUpdates: { 
      type: Boolean, 
      default: false 
    },
   
    feedbackText: { 
      type: String, 
      required: false 
    },
    featureSuggestions: { 
      type: [String], 
      required: false 
    },
    rating: { 
      type: Number, 
      required: false 
    },
    teamSize: {
      type: String,
      required: true,
    },
    region: {
      type: String,
      required: true,
    },
    ID_Organization: {
      type: String,
      required: function() { return this.status === 'AdministratorOrganization'; }, 
    },
    lastConnexion: {  // Nouveau champ ajouté ici
      type: Date,
      required: false,
    },
    resetPasswordToken: { type: String }, // Nouveau champ pour le jeton de réinitialisation

  },
  {
    timestamps: true,
  }
);

const modelName = 'Account';

export default model<I>(modelName, instance);