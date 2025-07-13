import { type RequestHandler } from 'express';
import joi from '../../utils/joi';
import jwt from '../../utils/jwt';
import crypt from '../../utils/crypt';
import { sendVerificationEmail } from '../../utils/nodemailer';
import Account from '../../models/Account';
import crypto from 'crypto';

const register: RequestHandler = async (req, res, next) => {
  try {
    const validationError = await joi.validate(
      {
        name: joi.instance.string().required(),
        surname: joi.instance.string().required(),
        organizationName: joi.instance.string().required(), // Ajouté
        email: joi.instance.string().email().required(),
        password: joi.instance.string().required(),
        role: joi.instance.string().valid('user', 'admin').default('user'), // Ajouté avec validation
        canInvite: joi.instance.boolean().default(false), // Ajouté
        isVerified: joi.instance.boolean().default(false), // Ajouté
        mfaEnabled: joi.instance.boolean().default(false), // Ajouté
      },
      req.body,
      { stripUnknown: true }
    );

    if (validationError) {
      return next(validationError);
    }

    const { name, surname, organizationName, email, password, role, canInvite, isVerified, mfaEnabled } = req.body;

    // Verify account email as unique
    const found = await Account.findOne({ email });

    if (found) {
      return next({
        statusCode: 400,
        message: 'An account already exists with this email',
      });
    }

    // Encrypt password
    const hash = await crypt.hash(password);

    // Generate 6-digit verification code
    const verificationCode = crypto.randomInt(100000, 999999).toString();
    const verificationCodeExpires = new Date(Date.now() + 30 * 1000); // 30 seconds

    // Create account
    const account = new Account({
      name,
      surname,
      organizationName,
      email,
      password: hash,
      role, // Ajouté
      isVerified,
      verificationCode,
      verificationCodeExpires,
      mfaEnabled, // Ajouté
      canInvite, // Ajouté
    });
    await account.save();

    // Send verification email
    await sendVerificationEmail(email, name, verificationCode);

    // Generate access token
    const token = jwt.signToken({ uid: account._id, role: account.role });

    // Exclude password and verificationCode from response
    const { password: _password, verificationCode: _code, verificationCodeExpires: _expires, ...data } = account.toObject();

    res.status(201).json({
      message: 'Signup successful! Please check your email for a verification code.',
      data,
      token,
    });
  } catch (error: any) {
    if (error.code === 11000) {
      return next({
        statusCode: 400,
        message: 'An account already exists with this email',
      });
    }
    next({
      statusCode: 500,
      message: 'Server error during signup',
      error: error.message,
    });
  }
};

export default register;