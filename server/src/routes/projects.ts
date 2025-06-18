import express from 'express';
import { getFtrackSession } from '../utils/session';

const router = express.Router();
const session = getFtrackSession();

router.get('/projects', async (req, res) => {
    try {
        console.log('Connected to ftrack');
        console.log('About to query projects');
        const response = await (await session).query('select id,name from Project');
        const projects = response.data;

        console.info("Listing " + projects.length + " projects");
        console.log(projects);

        res.json(projects);
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to fetch projects' });
    }
});


// Get detailed info for a single project by name
router.get('/projects/:projectName', async (req, res) => {
  try {
    const { projectName } = req.params;

    console.log(`Fetching details for Tasks with project id : ${projectName}`);

    const sessionInstance = await session;

    // Query for the project entity with more detailed fields
    const projectQuery = await (await session).query(
  `select 
    id, 
    name, 
    description, 
    status.name, 
    priority.name, 
    type.name, 
    start_date, 
    end_date, 
    time_logged, 
    bid, 
    bid_time_logged_difference, 
    created_at, 
    created_by.username, 
    project.name, 
    parent.name 
    from Task 
    where project.id is ${projectName}`
);

    const project = projectQuery.data[0];

    if (!project) {
      return res.status(404).json({ error: 'Tasks not found' });
    }

    console.log("Task found:", JSON.stringify(project, null, 2));

    res.json(projectQuery.data);

  } catch (err) {
    console.error('Error fetching project details:', err);
    res.status(500).json({ error: 'Failed to fetch project details' });
  }
});


export default router;
