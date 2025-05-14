This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

# Increasing Docker Disk Space in macOS

This guide explains how to fix the disk space issue encountered during Docker container builds.

## Option 1: Increase disk space through Docker Desktop UI (Recommended)

1. Open Docker Desktop
2. Click on the gear icon (⚙️) in the top-right corner to open Settings
3. Navigate to **Resources** > **Advanced**
4. Use the slider under **Disk image size** to increase the maximum disk space available
5. Click **Apply & Restart**

This is the simplest method and should resolve your "no space left on device" error.

## Option 2: Check and clean up Docker resources

If you don't want to increase the maximum disk space, you can clean up existing resources:

```bash
# View detailed space usage
docker system df -v

# Remove unused containers, networks, images, and build cache
docker system prune -a

# Force space reclamation
docker run --privileged --pid=host docker/desktop-reclaim-space
```

## Option 3: Use Docker Desktop CLI

If you prefer using the command line:

```bash
# Check current disk usage
cd ~/Library/Containers/com.docker.docker/Data/vms/0/data
ls -klsh Docker.raw

# The current disk size is shown, compared to maximum size
```

## Troubleshooting

If you're still experiencing issues:

1. Make sure you have enough free space on your host machine
2. Optimize your Dockerfile to reduce image size
3. Consider multi-stage builds to reduce the final image size

Remember that increasing the disk space is only necessary if you're working with large containers or many containers simultaneously.
