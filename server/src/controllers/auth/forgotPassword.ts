import { type RequestHandler } from 'express';
import joi from '../../utils/joi';
import jwt from '../../utils/jwt';
import { transporter } from '../../utils/nodemailer';
import Account from '../../models/Account';

const forgotPassword: RequestHandler = async (req, res): Promise<void> => {
  console.log('Requête de réinitialisation de mot de passe reçue :', req.body);

  const validationError = await joi.validate(
    {
      email: joi.instance.string().email().required(),
    },
    req.body,
    { stripUnknown: true }
  );

  if (validationError) {
    console.log('Erreur de validation :', validationError);
    res.status(400).json({
      message: validationError.message,
      error: validationError.message,
    });
    return;
  }

  const { email } = req.body;

  const account = await Account.findOne({ email });
  if (!account) {
    console.log('Aucun compte trouvé pour l\'email :', email);
    res.status(200).json({
      message: 'If the email address is associated with an account, an email has been sent with a reset link.',
    });
    return;
  }

  const resetToken = jwt.signToken({ uid: account._id }, '1h');
  console.log('Debug: Payload utilisé:', { uid: account._id });
  console.log('Debug: Signed token:', resetToken);
  console.log('Debug: Token expiration:', new Date(Date.now() + 60 * 60 * 1000));

  // Sauvegarder le jeton dans la base de données
  account.resetPasswordToken = resetToken;
  account.verificationTokenExpires = new Date(Date.now() + 60 * 60 * 1000); // Expire dans 1 heure
  console.log('Debug: Avant sauvegarde - resetPasswordToken:', account.resetPasswordToken);
  await account.save();
  console.log('Debug: Après sauvegarde - resetPasswordToken:', account.resetPasswordToken);
  console.log('Jeton de réinitialisation sauvegardé pour le compte :', account._id);

  const resetUrl = `${process.env.REACT_APP_FRONTEND_URL || 'http://localhost:3000'}/reset-password/${resetToken}`;
  const template = `
    <h2>Reset Your Password</h2>
    <p>Click the button below to reset your password:</p>
    <a href="${resetUrl}" style="display: inline-block; padding: 10px 20px; background-color: #4299e1; color: white; text-decoration: none; border-radius: 5px;">Reset Password</a>
    <p>This link will expire in 1 hour.</p>
    <p>For any questions, contact us at <a href="mailto:support@idara.com">support@idara.com</a>.</p>
  `;
  const mailOptions = {
    from: process.env.EMAIL_USER,
    to: email,
    subject: 'Reset Your IDARA Password',
    html: template,
  };

  try {
    await transporter.sendMail(mailOptions);
    console.log('Email de réinitialisation envoyé à :', email);
  } catch (error: any) {
    console.error('Erreur lors de l\'envoi de l\'email :', error);
    res.status(500).json({
      message: 'Failed to send reset email',
      error: error.message,
    });
    return;
  }

  res.status(200).json({
    message: 'If the email address is associated with an account, an email has been sent with a reset link.',
  });
};

export default forgotPassword;