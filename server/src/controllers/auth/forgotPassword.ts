// controllers/auth/forgotPassword.ts
import { type RequestHandler } from 'express';
import joi from '../../utils/joi';
import jwt from '../../utils/jwt';
import { transporter } from '../../utils/nodemailer'; // Assurez-vous que transporter est exporté dans nodemailer.ts
import Account from '../../models/Account';

const forgotPassword: RequestHandler = async (req, res, next) => {
    try {
      const validationError = await joi.validate(
        {
          email: joi.instance.string().email().required(),
        },
        req.body,
        { stripUnknown: true }
      );
  
      if (validationError) {
        return next(validationError);
      }
  
      const { email } = req.body;
  
      const account = await Account.findOne({ email });
      if (!account) {
        res.status(200).json({
          message: 'If the email address is associated with an account, an email has been sent with a reset link.',
        });
        return;
      }
  
      const resetToken = jwt.signToken({ uid: account._id }, '1h');
      const resetUrl = `${process.env.REACT_APP_FRONTEND_URL || 'http://localhost:3000'}/reset-password/${resetToken}`; // Pointe vers le frontend
  
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
  
      await transporter.sendMail(mailOptions);
      console.log('Reset password email sent');
  
      res.status(200).json({
        message: 'If the email address is associated with an account, an email has been sent with a reset link.',
      });
      return;
    } catch (error) {
      next(error);
    }
  };
  
  export default forgotPassword;