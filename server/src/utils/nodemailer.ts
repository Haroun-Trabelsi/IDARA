// utils/nodemailer.ts
import nodemailer from 'nodemailer';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
dotenv.config();

console.log('Debug: Nodemailer transporter initialized with user:', process.env.EMAIL_USER);

export const transporter = nodemailer.createTransport({ // Ajout de 'export'
  service: 'gmail',
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS,
  },
});
const getEmailTemplate = (templateName: string) => {
  const templatePath = path.join(__dirname, '..', 'utils', 'emailTemplates', `${templateName}.html`);
  return fs.readFileSync(templatePath, 'utf-8');
};

export const sendVerificationEmail = async (email: string, name: string, code: string) => {
  const template = getEmailTemplate('verificationEmail');
  const html = template
    .replace('{{name}}', name)
    .replace('{{code}}', code)
    .replace('{{greeting}}', `Hello ${name},`)
    .replace('{{message}}', 'Thank you for verifying your account with the following code:')
    .replace('{{expiry}}', 'This code expires in 24 hours.')
    .replace('{{contact}}', 'For any questions, contact us at <a href="mailto:support@idara.com">support@idara.com</a>.');

  const mailOptions = {
    from: process.env.EMAIL_USER,
    to: email,
    subject: 'Verify Your IDARA Account',
    html,
  };

  try {
    const info = await transporter.sendMail(mailOptions);
    console.log('Email sent:', info.response);
  } catch (error) {
    console.error('Error sending verification email:', error);
    throw new Error('Failed to send verification email');
  }
};

export const sendInvitationEmail = async (email: string, verificationLink: string, tempPassword: string, organizationName: string) => {
  const template = getEmailTemplate('invitationEmail');
  const loginUrl = `${process.env.APP_URL}/col/verify-invitation/${verificationLink}`;
  const html = template
    .replace('{{greeting}}', `Hello,`)
    .replace('{{message}}', `You have been invited to join the ${organizationName} team. Your temporary password is: <strong>${tempPassword}</strong>. Click the button below to accept the invitation:`)
    .replace('{{loginUrl}}', loginUrl)
    .replace('{{expiry}}', 'This invitation expires in 24 hours.')
    .replace('{{contact}}', 'For any questions, contact us at <a href="mailto:support@idara.com">support@idara.com</a>.');

  const mailOptions = {
    from: process.env.EMAIL_USER,
    to: email,
    subject: 'Invitation to Join IDARA Team',
    html,
  };

  try {
    const info = await transporter.sendMail(mailOptions);
    console.log('Invitation email sent:', info.response);
  } catch (error) {
    console.error('Error sending invitation email:', error);
    throw new Error('Failed to send invitation email');
  }
};