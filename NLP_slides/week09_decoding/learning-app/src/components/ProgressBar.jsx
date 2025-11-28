/**
 * ProgressBar Component
 * Shows overall progress at top of main area
 */

import { Box, LinearProgress, Typography } from '@mui/material';

const ProgressBar = ({ current, total }) => {
  const percentage = Math.round(((current + 1) / total) * 100);

  return (
    <Box sx={{ width: '100%', bgcolor: 'white', borderBottom: '1px solid', borderColor: 'divider', p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="body2" color="text.secondary">
          Progress: Slide {current + 1} of {total}
        </Typography>
        <Typography variant="body2" fontWeight="bold" color="primary">
          {percentage}%
        </Typography>
      </Box>
      <LinearProgress variant="determinate" value={percentage} sx={{ height: 8 }} />
    </Box>
  );
};

export default ProgressBar;
