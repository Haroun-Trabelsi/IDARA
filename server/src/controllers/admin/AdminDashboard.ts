
import { type RequestHandler } from 'express';
import Account from '../../models/Account';
import Feedback from '../../models/Feedback';
import { getFtrackSession } from '../../utils/session';

const getAdminDashboardData: RequestHandler = async (req, res, next) => {
  try {
    const sessionInstance = await getFtrackSession();

    // Récupérer les comptes
    const accounts = await Account.find({ role: 'user' }).select('organizationName email _id lastConnexion createdAt');
    const organizations = [...new Set(accounts.map(acc => acc.organizationName))].length; // Nombre unique d'organisations
    const totalUsers = accounts.length;

    // Récupérer tous les projets depuis Ftrack
    let totalProjects = 0;
    let projectMetrics: any[] = [];
    try {
      const projectResponse = await sessionInstance.query('select id, name, status from Project');
      const projects = projectResponse.data;
      totalProjects = projects.length;

      // Calculer les métriques des projets par organisation
      projectMetrics = await Promise.all(
        [...new Set(accounts.map(acc => acc.organizationName))].map(async (orgName) => {
          // Alternative : Récupérer les projets liés à l'organisation via les utilisateurs
          const orgUsers = await sessionInstance.query(
            `select id, username from User where email in (${accounts
              .filter(acc => acc.organizationName === orgName)
              .map(acc => `"${acc.email}"`)
              .join(', ')})`
          );
          const userIds = orgUsers.data.map((user: any) => user.id);

          // Récupérer les projets où ces utilisateurs sont impliqués
          const orgProjects = await sessionInstance.query(
            `select id, name, status from Project where assignments any (resource_id in (${userIds.join(', ')}))`
          );
          const projectsData = orgProjects.data || [];
          const activeProjects = projectsData.filter((p: any) => p.status === 'active').length;
          const completedProjects = projectsData.filter((p: any) => p.status === 'completed').length;

          // Récupérer le nombre total de tâches pour ces projets
          const totalTasks = await sessionInstance.query(
            `select count() from Task where project_id in (${projectsData.map((p: any) => `"${p.id}"`).join(', ')})`
          );

          return {
            organizationName: orgName,
            totalProjects: projectsData.length,
            activeProjects,
            completedProjects,
            totalTasks: totalTasks.data[0]?.count || 0,
          };
        })
      );
    } catch (error) {
      console.error('Error fetching Ftrack projects:', error);
      // Continuer avec des données partielles
      projectMetrics = [...new Set(accounts.map(acc => acc.organizationName))].map(orgName => ({
        organizationName: orgName,
        totalProjects: 0,
        activeProjects: 0,
        completedProjects: 0,
        totalTasks: 0,
      }));
    }

    // Calculer la croissance des utilisateurs par mois
    const growthData = accounts.reduce((acc, curr) => {
      const month = new Date(curr.createdAt).toLocaleString('default', { month: 'short', year: 'numeric' });
      acc[month] = (acc[month] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    const organizationGrowthData = Object.keys(growthData)
      .map(month => ({
        month,
        organizations: growthData[month],
        users: growthData[month] * 8, // Approximation basée sur un facteur
      }))
      .slice(-6); // Derniers 6 mois

    // Récupérer les ratings depuis le modèle Feedback
    const feedbacks = await Feedback.find().populate('accountId', 'email organizationName');
    const ratingDistribution = [1, 2, 3, 4, 5].map(rating => ({
      rating,
      count: feedbacks.filter(f => f.rating === rating).length,
      percentage: (feedbacks.filter(f => f.rating === rating).length / (feedbacks.length || 1)) * 100,
    }));

    // Formatter les données des organisations avec lastConnexion et feedbacks
    const organizationsList = await Promise.all(
      accounts.map(async (acc) => {
        const orgFeedbacks = await Feedback.find({ accountId: acc._id }).select('rating feedbackText featureSuggestions createdAt');
        return {
          id: acc._id.toString(),
          name: acc.organizationName,
          email: acc.email,
          projects: projectMetrics.find(pm => pm.organizationName === acc.organizationName)?.totalProjects || 0,
          users: accounts.filter(a => a.organizationName === acc.organizationName).length,
          status: 'active', // Simplifié, ajuste selon ta logique
          joinDate: acc.createdAt.toISOString().split('T')[0],
          lastActive: acc.lastConnexion ? new Date(acc.lastConnexion).toLocaleString('fr-FR', { dateStyle: 'short', timeStyle: 'short' }) : 'No connexion',
          rating: orgFeedbacks.length ? orgFeedbacks.reduce((sum, f) => sum + f.rating, 0) / orgFeedbacks.length : 0,
          feedbacks: orgFeedbacks.map(f => ({
            rating: f.rating,
            feedbackText: f.feedbackText || '',
            featureSuggestions: f.featureSuggestions || [],
            createdAt: f.createdAt.toISOString().split('T')[0],
          })),
        };
      })
    );

    res.status(200).json({
      totalOrganizations: organizations,
      totalProjects,
      totalUsers,
      organizationGrowthData,
      ratingDistribution,
      organizationsList,
      projectMetrics,
    });
  } catch (error) {
    console.error('Error in getAdminDashboardData:', error);
    //res.status(500).json({ message: 'Internal server error', error: error.message });
  }
};

// Récupérer les feedbacks par utilisateur
const getFeedbacks: RequestHandler = async (req, res, next) => {
  try {
    const feedbacks = await Feedback.find().populate('accountId', 'email organizationName');
    const formattedFeedbacks = feedbacks.map(f => ({
      id: (f as any)._id.toString(),
      accountEmail: f.accountId ? (f.accountId as any).email : 'Unknown',
      organizationName: f.accountId ? (f.accountId as any).organizationName : 'Unknown',
      rating: f.rating,
      feedbackText: f.feedbackText || '',
      featureSuggestions: f.featureSuggestions || [],
      createdAt: f.createdAt.toISOString().split('T')[0],
    }));
    res.status(200).json(formattedFeedbacks);
  } catch (error) {
    next(error);
  }
};

// Envoyer un email
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

export { getAdminDashboardData, getFeedbacks, sendEmail };
