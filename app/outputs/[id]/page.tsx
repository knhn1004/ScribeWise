'use client';

import { useState, useEffect } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import {
	FileText,
	Download,
	BookOpen,
	Brain,
	Film,
	File,
	ArrowLeft,
	AlertCircle,
	Loader,
	ExternalLink,
	ChevronLeft,
} from 'lucide-react';
import { motion } from 'framer-motion';
import {
	Card,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';

// API base URL - can be moved to env variable
const API_BASE_URL = 'http://localhost:8000';

// Define types for our data
interface VideoData {
	video_info?: {
		title: string;
	};
	transcription?: {
		text: string;
	};
	outputs?: {
		summary_path?: string;
		notes_path?: string;
		flashcards_path?: string;
		mindmap_path?: string;
		mindmap_image_url?: string;
		mindmap_image_path?: string;
		anki_path?: string;
	};
}

interface PDFData {
	title?: string;
	notes_path?: string;
	flashcards_path?: string;
	mindmap_path?: string;
	anki_path?: string;
}

interface OutputItem {
	id: string;
	title: string;
	description: string;
	icon: React.ReactNode;
	path?: string;
	content?: string;
	download: boolean;
	special?: boolean;
	isText?: boolean;
	color: string;
}

export default function OutputsPage() {
	const params = useParams();
	const searchParams = useSearchParams();
	const id = params.id as string;
	const type = searchParams.get('type') || 'video'; // 'video' or 'pdf'

	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [data, setData] = useState<VideoData | PDFData | null>(null);
	const [title, setTitle] = useState<string>('');
	const [activeOutput, setActiveOutput] = useState<OutputItem | null>(null);
	const [showDebug, setShowDebug] = useState(false);

	useEffect(() => {
		async function fetchData() {
			try {
				setLoading(true);
				setError(null);

				// Fetch data based on type
				const endpoint = type === 'pdf' ? `/pdfs/${id}` : `/videos/${id}`;
				const response = await fetch(`${API_BASE_URL}${endpoint}`);

				if (!response.ok) {
					throw new Error(`Failed to fetch data: ${response.statusText}`);
				}

				const result = await response.json();
				setData(result);

				// Set title based on the type of content
				if (type === 'pdf') {
					setTitle(result.title || 'PDF Document');
				} else {
					setTitle(result.video_info?.title || 'Video Content');
				}
			} catch (err) {
				console.error('Error fetching data:', err);
				setError(
					err instanceof Error ? err.message : 'An unknown error occurred'
				);
			} finally {
				setLoading(false);
			}
		}

		if (id) {
			fetchData();
		}
	}, [id, type]);

	// Get available outputs based on content type
	const getOutputs = (): OutputItem[] => {
		if (!data) return [];

		const outputs: OutputItem[] = [];

		if (type === 'pdf') {
			const pdfData = data as PDFData;
			// PDF outputs
			if (pdfData.notes_path) {
				outputs.push({
					id: 'notes',
					title: 'Study Notes',
					description:
						'Comprehensive study notes generated from the PDF content',
					icon: <BookOpen className="h-8 w-8" />,
					path: pdfData.notes_path,
					download: true,
					color: 'blue',
				});
			}

			if (pdfData.flashcards_path) {
				outputs.push({
					id: 'flashcards',
					title: 'Flashcards',
					description: 'Study flashcards to test your knowledge',
					icon: <FileText className="h-8 w-8" />,
					path: pdfData.flashcards_path,
					download: true,
					color: 'green',
				});
			}

			if (pdfData.mindmap_path) {
				outputs.push({
					id: 'mindmap-md',
					title: 'Mindmap (Markdown)',
					description:
						'Visual representation of content structure in Markdown format',
					icon: <Brain className="h-8 w-8" />,
					path: pdfData.mindmap_path,
					download: true,
					color: 'purple',
				});
			}

			// Add mindmap image if available
			outputs.push({
				id: 'mindmap-image',
				title: 'Mindmap Image',
				description: 'Visual mindmap as SVG or PNG image',
				icon: <Brain className="h-8 w-8" />,
				path: `/pdf-mindmap-image/${id}`,
				download: true,
				special: true,
				color: 'indigo',
			});

			if (pdfData.anki_path) {
				outputs.push({
					id: 'anki',
					title: 'Anki Deck',
					description: 'Importable Anki flashcard deck',
					icon: <File className="h-8 w-8" />,
					path: pdfData.anki_path,
					download: true,
					color: 'orange',
				});
			}
		} else {
			const videoData = data as VideoData;
			// Video outputs from the outputs field
			const videoOutputs = videoData.outputs || {};

			if (videoOutputs.summary_path) {
				outputs.push({
					id: 'summary',
					title: 'Summary',
					description: 'Concise summary of the video content',
					icon: <FileText className="h-8 w-8" />,
					path: videoOutputs.summary_path,
					download: true,
					color: 'blue',
				});
			}

			if (videoOutputs.notes_path) {
				outputs.push({
					id: 'notes',
					title: 'Study Notes',
					description: 'Detailed notes from the video content',
					icon: <BookOpen className="h-8 w-8" />,
					path: videoOutputs.notes_path,
					download: true,
					color: 'green',
				});
			}

			if (videoOutputs.flashcards_path) {
				outputs.push({
					id: 'flashcards',
					title: 'Flashcards',
					description: 'Study flashcards from the video',
					icon: <FileText className="h-8 w-8" />,
					path: videoOutputs.flashcards_path,
					download: true,
					color: 'yellow',
				});
			}

			if (videoOutputs.mindmap_path) {
				outputs.push({
					id: 'mindmap-md',
					title: 'Mindmap (Markdown)',
					description: 'Visual representation of content in Markdown format',
					icon: <Brain className="h-8 w-8" />,
					path: videoOutputs.mindmap_path,
					download: true,
					color: 'purple',
				});
			}

			// Add mindmap image if available (handles both URL formats)
			if (videoOutputs.mindmap_image_url || videoOutputs.mindmap_image_path) {
				outputs.push({
					id: 'mindmap-image',
					title: 'Mindmap Image',
					description: 'Visual mindmap as SVG or PNG image',
					icon: <Brain className="h-8 w-8" />,
					path: videoOutputs.mindmap_image_url || `/mindmap-image/${id}`,
					download: true,
					special: true,
					color: 'indigo',
				});
			}

			if (videoOutputs.anki_path) {
				outputs.push({
					id: 'anki',
					title: 'Anki Deck',
					description: 'Importable Anki flashcard deck',
					icon: <File className="h-8 w-8" />,
					path: videoOutputs.anki_path,
					download: true,
					color: 'orange',
				});
			}

			// Add the video transcription if available
			if (videoData.transcription?.text) {
				outputs.push({
					id: 'transcription',
					title: 'Transcription',
					description: 'Full text transcription of the video',
					icon: <Film className="h-8 w-8" />,
					content: videoData.transcription.text,
					download: false,
					isText: true,
					color: 'red',
				});
			}
		}

		return outputs;
	};

	// Handle downloading content
	const handleDownload = (path: string, filename: string) => {
		// For special paths that need the API_BASE_URL
		let fullPath = path;

		if (path.startsWith('/app/outputs/')) {
			// Fix the path to use the correct format for the API
			fullPath = `${API_BASE_URL}/outputs/${path.replace('/app/outputs/', '')}`;
		} else if (path.startsWith('/outputs/')) {
			fullPath = `${API_BASE_URL}${path}`;
		} else if (path.startsWith('/')) {
			fullPath = `${API_BASE_URL}${path}`;
		}

		console.log('Download path:', path, '→', fullPath);

		// Create a link element and trigger download
		const link = document.createElement('a');
		link.href = fullPath;
		link.download = filename || 'download';
		document.body.appendChild(link);
		link.click();
		document.body.removeChild(link);
	};

	// Determine file extension from path or fallback to txt
	const getFileExtension = (path: string) => {
		if (!path) return 'txt';
		const parts = path.split('.');
		return parts.length > 1 ? parts[parts.length - 1] : 'txt';
	};

	// Get filename for download
	const getFilename = (output: OutputItem) => {
		const extension = getFileExtension(output.path || '');
		return `${title.replace(/\s+/g, '_')}_${output.id}.${extension}`;
	};

	// Get color class based on output color
	const getColorClass = (
		color: string,
		element: 'text' | 'bg' | 'border' | 'ring'
	) => {
		const colorMap: Record<string, Record<string, string>> = {
			blue: {
				text: 'text-blue-500',
				bg: 'bg-blue-500/10',
				border: 'border-blue-500/20',
				ring: 'ring-blue-500/20',
			},
			green: {
				text: 'text-green-500',
				bg: 'bg-green-500/10',
				border: 'border-green-500/20',
				ring: 'ring-green-500/20',
			},
			yellow: {
				text: 'text-yellow-500',
				bg: 'bg-yellow-500/10',
				border: 'border-yellow-500/20',
				ring: 'ring-yellow-500/20',
			},
			purple: {
				text: 'text-purple-500',
				bg: 'bg-purple-500/10',
				border: 'border-purple-500/20',
				ring: 'ring-purple-500/20',
			},
			indigo: {
				text: 'text-indigo-500',
				bg: 'bg-indigo-500/10',
				border: 'border-indigo-500/20',
				ring: 'ring-indigo-500/20',
			},
			orange: {
				text: 'text-orange-500',
				bg: 'bg-orange-500/10',
				border: 'border-orange-500/20',
				ring: 'ring-orange-500/20',
			},
			red: {
				text: 'text-red-500',
				bg: 'bg-red-500/10',
				border: 'border-red-500/20',
				ring: 'ring-red-500/20',
			},
		};

		return colorMap[color]?.[element] || '';
	};

	if (loading) {
		return (
			<div className="container mx-auto py-12 px-4">
				<div className="mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
					<div>
						<Skeleton className="h-10 w-64 mb-2" />
						<Skeleton className="h-5 w-48" />
					</div>
					<Skeleton className="h-10 w-40" />
				</div>

				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
					{[1, 2, 3, 4, 5, 6].map(i => (
						<Card key={i} className="overflow-hidden">
							<CardHeader className="pb-3">
								<Skeleton className="h-8 w-8 mb-2 rounded-full" />
								<Skeleton className="h-6 w-32 mb-1" />
								<Skeleton className="h-4 w-full" />
							</CardHeader>
							<CardFooter className="pt-3 flex justify-end gap-2">
								<Skeleton className="h-9 w-28" />
								<Skeleton className="h-9 w-20" />
							</CardFooter>
						</Card>
					))}
				</div>
			</div>
		);
	}

	if (error) {
		return (
			<div className="container mx-auto py-12 px-4">
				<Alert className="bg-red-500/10 border-red-500/20 text-red-500 mb-4">
					<AlertCircle className="h-4 w-4" />
					<AlertDescription>{error}</AlertDescription>
				</Alert>
				<Button asChild>
					<Link href="/dashboard">
						<ArrowLeft className="mr-2 h-4 w-4" />
						Back to Dashboard
					</Link>
				</Button>
			</div>
		);
	}

	const outputs = getOutputs();

	console.log('API_BASE_URL:', API_BASE_URL);
	console.log(
		'Output paths:',
		outputs.map(o => ({
			id: o.id,
			path: o.path,
			fullPath: o.path?.startsWith('/') ? `${API_BASE_URL}${o.path}` : o.path,
		}))
	);

	return (
		<div className="container mx-auto py-12 px-4">
			<motion.div
				className="mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4"
				initial={{ opacity: 0, y: -20 }}
				animate={{ opacity: 1, y: 0 }}
				transition={{ duration: 0.5 }}
			>
				<div>
					<h1 className="text-3xl font-bold tracking-tight">{title}</h1>
					<p className="text-muted-foreground mt-1">
						{type === 'pdf' ? 'PDF' : 'Video'} learning materials and outputs
					</p>
				</div>
				<Button asChild variant="outline" className="group">
					<Link href="/dashboard">
						<ChevronLeft className="mr-2 h-4 w-4 transition-transform group-hover:-translate-x-1" />
						Back to Dashboard
					</Link>
				</Button>
			</motion.div>

			{outputs.length === 0 ? (
				<Alert className="bg-amber-500/10 border-amber-500/20">
					<AlertCircle className="h-5 w-5 text-amber-500" />
					<AlertDescription className="text-sm">
						No outputs are available for this content yet. Processing may still
						be in progress.
					</AlertDescription>
				</Alert>
			) : (
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
					{outputs.map((output, index) => (
						<motion.div
							key={output.id}
							initial={{ opacity: 0, y: 20 }}
							animate={{ opacity: 1, y: 0 }}
							transition={{ duration: 0.5, delay: index * 0.1 }}
						>
							<Card
								className={`overflow-hidden border transition-all duration-300 hover:shadow-md ${
									activeOutput?.id === output.id
										? `ring-2 ${getColorClass(output.color, 'ring')}`
										: ''
								}`}
							>
								<CardHeader className="pb-3">
									<div
										className={`mb-2 p-2 rounded-full w-fit ${getColorClass(
											output.color,
											'bg'
										)}`}
									>
										<div className={getColorClass(output.color, 'text')}>
											{output.icon}
										</div>
									</div>
									<CardTitle>{output.title}</CardTitle>
									<CardDescription>{output.description}</CardDescription>
								</CardHeader>

								<CardFooter className="pt-3 flex justify-end gap-2">
									<motion.div
										whileHover={{ scale: 1.05 }}
										whileTap={{ scale: 0.95 }}
									>
										<Button
											variant="default"
											className={`border ${getColorClass(
												output.color,
												'text'
											)} ${getColorClass(
												output.color,
												'border'
											)} hover:${getColorClass(
												output.color,
												'bg'
											)} bg-background`}
											onClick={() => {
												setActiveOutput(output);
												if (output.isText) {
													// For text content, create a downloadable file from the content
													const blob = new Blob([output.content || ''], {
														type: 'text/plain',
													});
													const url = URL.createObjectURL(blob);
													handleDownload(
														url,
														`${title.replace(/\s+/g, '_')}_${output.id}.txt`
													);
													setTimeout(() => URL.revokeObjectURL(url), 100);
												} else if (output.download) {
													// For downloadable outputs
													handleDownload(
														output.path || '',
														getFilename(output)
													);
												}
											}}
										>
											<Download className="mr-2 h-4 w-4" />
											Download
										</Button>
									</motion.div>

									{/* View button for paths that can be viewed in browser */}
									{!output.isText && output.path && (
										<Button
											variant="outline"
											onClick={() => {
												let fullPath = output.path || '';

												if (fullPath.startsWith('/app/outputs/')) {
													// Fix the path to use the correct format for the API
													fullPath = `${API_BASE_URL}/outputs/${fullPath.replace(
														'/app/outputs/',
														''
													)}`;
												} else if (fullPath.startsWith('/outputs/')) {
													fullPath = `${API_BASE_URL}${fullPath}`;
												} else if (fullPath.startsWith('/')) {
													fullPath = `${API_BASE_URL}${fullPath}`;
												}

												console.log('View path:', output.path, '→', fullPath);
												window.open(fullPath, '_blank');
											}}
										>
											<ExternalLink className="mr-2 h-4 w-4" />
											View
										</Button>
									)}
								</CardFooter>
							</Card>
						</motion.div>
					))}
				</div>
			)}

			{outputs.length > 0 && (
				<motion.div
					className="mt-12"
					initial={{ opacity: 0 }}
					animate={{ opacity: 1 }}
					transition={{ duration: 0.5, delay: 0.5 }}
				>
					<Separator className="mb-6" />
					<div className="text-center">
						<h2 className="text-xl font-semibold mb-2">
							Need more learning materials?
						</h2>
						<p className="text-muted-foreground mb-4">
							Upload more content to generate additional study materials
						</p>
						<div className="flex justify-center gap-4">
							<Button asChild>
								<Link href="/upload">Upload New Content</Link>
							</Button>
							<Button
								variant="outline"
								className="border-primary text-primary hover:bg-primary/10"
								onClick={() => setShowDebug(!showDebug)}
							>
								{showDebug ? 'Hide Debug' : 'Show Debug'}
							</Button>
						</div>
					</div>
				</motion.div>
			)}

			{showDebug && (
				<div className="mt-8 p-4 bg-muted/50 rounded-md overflow-auto max-h-96">
					<h3 className="text-lg font-semibold mb-2">Debug Info</h3>
					<pre className="text-xs">
						{JSON.stringify(
							{
								API_BASE_URL,
								id,
								type,
								title,
								outputPaths: outputs.map(o => ({
									id: o.id,
									path: o.path,
									fullPath: o.path?.startsWith('/')
										? `${API_BASE_URL}${o.path}`
										: o.path,
								})),
							},
							null,
							2
						)}
					</pre>
				</div>
			)}
		</div>
	);
}
