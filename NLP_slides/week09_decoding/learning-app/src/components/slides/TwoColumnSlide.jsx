/**
 * Two-Column Slide - Type 2
 * Side-by-side content comparison
 */

import { Grid, Card, CardContent, Typography, Box } from '@mui/material';
import { motion } from 'framer-motion';

const TwoColumnSlide = ({ title, leftTitle, leftContent, rightTitle, rightContent }) => {
  return (
    <motion.div
      initial={{ x: -20, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <Box>
        {title && (
          <Typography variant="h4" color="primary" gutterBottom fontWeight="bold" sx={{ mb: 3 }}>
            {title}
          </Typography>
        )}
        <Grid container spacing={3}>
          <Grid item xs={6}>
            <Card elevation={2} sx={{ height: '100%', bgcolor: 'secondary.light' }}>
              <CardContent>
                <Typography variant="h5" color="primary" gutterBottom fontWeight="600">
                  {leftTitle}
                </Typography>
                <Typography variant="body1" color="text.primary">
                  {leftContent}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={6}>
            <Card elevation={2} sx={{ height: '100%', bgcolor: 'primary.light', color: 'white' }}>
              <CardContent>
                <Typography variant="h5" color="white" gutterBottom fontWeight="600">
                  {rightTitle}
                </Typography>
                <Typography variant="body1" color="white">
                  {rightContent}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </motion.div>
  );
};

export default TwoColumnSlide;
