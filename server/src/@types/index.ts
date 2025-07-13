export interface Account {
  name: string;
  surname: string;
  email: string;
  password: string;
  role: 'user' | 'admin';
  isVerified: boolean;
  verificationCode?: string;
  verificationCodeExpires?: Date;
  mfaSecret?: string ;
  mfaEnabled: boolean ;
}
