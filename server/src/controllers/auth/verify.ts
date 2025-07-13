import { type RequestHandler, type Request, type Response, type NextFunction } from 'express'
import Account from '../../models/Account'

const verify: RequestHandler = async (req: Request, res: Response, next: NextFunction) => {
  try {
    let token: string | undefined
    let code: string | undefined
    let email: string | undefined

    if (req.method === 'GET') {
      token = req.params.token
    } else if (req.method === 'POST') {
      ({ code, email } = req.body)
    }

    if (!code || !email) {
      return next({
        statusCode: 400,
        message: 'Code and email are required',
      })
    }

    const account = await Account.findOne({ verificationCode: code, email })

    if (!account || (account.verificationCodeExpires && account.verificationCodeExpires < new Date())) {
      return next({
        statusCode: 400,
        message: 'Invalid or expired verification code',
      })
    }

    account.isVerified = true
    account.verificationCode = undefined
    account.verificationCodeExpires = undefined
    await account.save()

    res.status(200).json({ message: 'Email verified successfully! You can now log in.' })
  } catch (error: any) {
    console.log('Verification error:', error) // Débogage
    return next({
      statusCode: 500,
      message: 'Server error during verification',
      error: error.message,
    })
  }
}

export default verify