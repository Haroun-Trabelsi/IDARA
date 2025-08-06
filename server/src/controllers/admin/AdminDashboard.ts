import { type RequestHandler } from 'express';
import Account from '../../models/Account';

// Récupérer les données du tableau de bord admin
const getAdminDashboardData: RequestHandler = async (req, res, next) => {
  try {
    const accounts = await Account.find({ role: 'user' }).select('organizationName email');
    const organizations = [...new Set(accounts.map(acc => acc.organizationName))].length; // Nombre unique d'organisations
    const totalUsers = accounts.length;

    // Simuler des projets statiques (remplace par une vraie logique si tu as un modèle Project)
    const activeProjects = 89; // Statique comme demandé

    // Calculer les utilisateurs par organisation pour la croissance (simplifié)
    const growthData = accounts.reduce((acc, curr) => {
      const month = new Date(curr.createdAt).toLocaleString('default', { month: 'short' });
      acc[month] = (acc[month] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    const organizationGrowthData = Object.keys(growthData).map(month => ({
      month,
      organizations: growthData[month],
      users: growthData[month] * 8 // Approximation basée sur un facteur
    })).slice(-6); // Derniers 6 mois

    // Ratings statiques (remplace par une logique réelle si tu as des ratings)
    const ratingDistribution = [
      { rating: 5, count: 65, percentage: 44.2 },
      { rating: 4, count: 48, percentage: 32.7 },
      { rating: 3, count: 22, percentage: 15.0 },
      { rating: 2, count: 8, percentage: 5.4 },
      { rating: 1, count: 4, percentage: 2.7 }
    ];

    res.status(200).json({
      totalOrganizations: organizations,
      activeProjects,
      totalUsers,
      organizationGrowthData,
      ratingDistribution,
      organizationsList: accounts.map(acc => ({
        id: acc._id.toString(),
        name: acc.organizationName,
        email: acc.email
      }))
    });
  } catch (error) {
    next(error);
  }
};

// Envoyer un email (exemple avec un service comme Nodemailer, à implémenter)
const sendEmail: RequestHandler = async (req, res, next) => {
  try {
    const { subject, message, target } = req.body;
    const accounts = await Account.find();

    let targetEmails;
    if (target === 'all') {
      targetEmails = accounts.map(acc => acc.email);
    } else if (target) {
      const targetOrg = await Account.findOne({ organizationName: target }).select('email');
      targetEmails = targetOrg ? [targetOrg.email] : [];
    } else {
      return next({ statusCode: 400, message: 'Target is required' });
    }

    // Logique d'envoi d'email (à implémenter avec Nodemailer ou un service email)
    console.log(`Sending email to ${targetEmails.length} recipients: Subject: ${subject}, Message: ${message}`);

    res.status(200).json({ message: 'Email sending initiated', recipients: targetEmails.length });
  } catch (error) {
    next(error);
  }
};

export { getAdminDashboardData, sendEmail };