import { type RequestHandler, type Request, type Response, type NextFunction } from 'express';
import Account from '../../models/Account';
import crypto from 'crypto';
import { sendVerificationEmail } from '../../utils/nodemailer';

// Interface pour les erreurs personnalisées
interface CustomError {
  statusCode: number;
  message: string;
  error?: string;
}

// Interface pour le corps de la requête de vérification (POST)
interface VerifyBody {
  code: string;
  email: string;
}

// Interface pour le corps de la requête de renvoi de code
interface ResendBody {
  email: string;
}

// Contrôleur pour la vérification de l'email
const verify: RequestHandler<{ token?: string }, any, VerifyBody, {}> = async (
  req: Request<{ token?: string }, any, VerifyBody, {}>,
  res: Response,
  next: NextFunction
) => {
  try {
    let token: string | undefined;
    let code: string | undefined;
    let email: string | undefined;

    if (req.method === 'GET') {
      token = req.params.token;
    } else if (req.method === 'POST') {
      ({ code, email } = req.body);
    }

    if (!code || !email) {
      console.log('Invalid request:', req.body);
      return next({
        statusCode: 400,
        message: 'Code and email are required',
      } as CustomError);
    }

    console.log('Verifying code:', { code, email });
    const account = await Account.findOne({ 
      verificationCode: code, 
      email: { $regex: new RegExp(`^${email}$`, 'i') }
    });

    if (!account) {
      console.log('No account found with code and email:', { code, email });
      return next({
        statusCode: 400,
        message: 'Invalid or expired verification code',
      } as CustomError);
    }

    if (account.verificationCodeExpires && account.verificationCodeExpires < new Date()) {
      console.log('Code expired:', { expires: account.verificationCodeExpires });
      return next({
        statusCode: 400,
        message: 'Invalid or expired verification code',
      } as CustomError);
    }

    account.isVerified = true;
    account.verificationCode = undefined;
    account.verificationCodeExpires = undefined;
    await account.save();

    res.status(200).json({ message: 'Email verified successfully!' });
  } catch (error: any) {
    console.log('Verification error:', error);
    return next({
      statusCode: 500,
      message: 'Server error during verification',
      error: error.message,
    } as CustomError);
  }
};

// Contrôleur pour renvoyer le code de vérification
const resendVerification: RequestHandler<{}, any, ResendBody, {}> = async (
  req: Request<{}, any, ResendBody, {}>,
  res: Response,
  next: NextFunction
) => {
  try {
    const { email } = req.body;
    if (!email) {
      console.log('Invalid request:', req.body);
      return next({
        statusCode: 400,
        message: 'Email is required.',
      } as CustomError);
    }

    console.log('Resending code for:', email);
    const user = await Account.findOne({ email: { $regex: new RegExp(`^${email}$`, 'i') }, isVerified: false });
    if (!user) {
      console.log('User not found or already verified:', { email });
      return next({
        statusCode: 404,
        message: 'User not found or already verified.',
      } as CustomError);
    }

    const verificationCode = crypto.randomInt(100000, 999999).toString();
    const verificationCodeExpires = new Date(Date.now() + 30 * 1000);
    user.verificationCode = verificationCode;
    user.verificationCodeExpires = verificationCodeExpires;
    await user.save();

    console.log('Sending email with code:', verificationCode);
    await sendVerificationEmail(email, user.name, verificationCode);

    res.status(200).json({ message: 'Verification code resent successfully.' });
  } catch (error: any) {
    console.log('Resend error:', error);
    return next({
      statusCode: 500,
      message: 'Failed to resend verification code',
      error: error.message,
    } as CustomError);
  }
};

export { verify, resendVerification };