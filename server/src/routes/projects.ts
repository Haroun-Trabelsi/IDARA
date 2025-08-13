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


router.get('/projects/:projectName', async (req, res) => {
  try {
    const { projectName } = req.params;
    console.log(`📋 Fetching tasks for project ID: ${projectName}`);

    const sessionInstance = await session;

    // Query all tasks for the given project, including notes
    const taskQuery = await sessionInstance.query(`
      select
        id,
        name,
        description,
        type.name,
        status.name,
        end_date,
        bid,
        time_logged,
        parent.name,
        parent.parent.name,
        assignments.resource.id,
        notes.id
      from Task
      where project.id is ${projectName}
    `);

    const rawTasks = taskQuery.data;

    // Extract user IDs
    const userIds = [
      ...new Set(
        rawTasks.flatMap(t =>
          t.assignments?.map(a => a.resource?.id).filter(Boolean) || []
        )
      )
    ];

    // Get user names
    let usersById = {};
    if (userIds.length > 0) {
      const userQuery = await sessionInstance.query(`
        select id, first_name, last_name, username 
        from User 
        where id in (${userIds.join(',')})
      `);
      usersById = Object.fromEntries(
        userQuery.data.map(u => [
          u.id,
          `${u.first_name || ''} ${u.last_name || ''}`.trim() || u.username
        ])
      );
    }

    const tasks = rawTasks.map((task, index) => {
      const assignees = task.assignments
        ?.map(a => usersById[a.resource?.id])
        .filter(Boolean)
        .join(', ') || 'Unassigned';

      return {
        id: task.id,
        number: index + 1,
        type: `Task (${task.type?.name || 'Unknown'})`,
        status: task.status?.name?.toLowerCase() || 'pending',
        assignee: assignees,
        sequence: task.parent?.parent?.name || 'Unassigned',
        taskName: task.parent?.name || 'Unknown',
        description: task.description || '',
        notes_count: task.notes?.length || 0,
        dueDate: task.end_date ? new Date(task.end_date).toISOString().split('T')[0] : null,
        bidHours: task.bid ? task.bid / 3600 : 0,
        actualHours: task.time_logged ? task.time_logged / 3600 : 0,
        deltaHours: ((task?.time_logged || 0) - (task?.bid || 0)) / 3600
      };
    });

    res.json(tasks);
  } catch (err) {
    console.error('❌ Failed to fetch project tasks:', err);
    res.status(500).json({ error: 'Failed to fetch project tasks' });
  }
});



router.get('/task/:taskId/components', async (req, res) => {
  const { taskId } = req.params;
  try {
    const sessionInstance = await session;

    // 1. Get the shot ID from the task's parent
    const taskQuery = await sessionInstance.query(`
      select parent.id from Task where id is ${taskId}
    `);

    const shotId = taskQuery.data?.[0]?.parent?.id;
    if (!shotId) {
      return res.status(404).json({ error: 'Shot context not found for task' });
    }

    // 2. Fetch AssetVersions under Shot
    const shotVersionsQuery = await sessionInstance.query(`
      select
        asset.name,
        components.name,
        components.file_type,
        components.component_locations.url,
        date
      from AssetVersion
      where asset.parent.id is ${shotId}
      order by date desc
    `);

    // 3. Fetch AssetVersions directly under Task
    const taskVersionsQuery = await sessionInstance.query(`
      select
        asset.name,
        components.name,
        components.file_type,
        components.component_locations.url,
        date
      from AssetVersion
      where task.id is ${taskId}
      order by date desc
    `);

    const extractVideos = (versions: any[]) =>
      versions
        .map((av: any) => ({
          name: av.components?.[0]?.name || av.asset?.name || 'Unnamed',
          fileType: av.components?.[0]?.file_type || '',
          url: av.components?.[0]?.component_locations?.[0]?.url || '',
          date: av.date,
        }))
        .filter(v => v.url);

    const videos = [
      ...extractVideos(shotVersionsQuery.data),
      ...extractVideos(taskVersionsQuery.data),
    ];

    res.json(videos);

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to fetch video components from task and shot context' });
  }
});




export default router;
