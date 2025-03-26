import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle, BookOpen, Brain, Sparkles } from "lucide-react"

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Navigation */}
      <header className="border-b">
        <div className="container mx-auto max-w-7xl px-4 flex h-16 items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl">
            <Sparkles className="h-5 w-5 text-primary" />
            <span>ScribeWise</span>
          </div>
          <nav className="hidden md:flex gap-6">
            <Link href="#features" className="text-sm font-medium hover:underline underline-offset-4">
              Features
            </Link>
            <Link href="#how-it-works" className="text-sm font-medium hover:underline underline-offset-4">
              How It Works
            </Link>
            <Link href="#pricing" className="text-sm font-medium hover:underline underline-offset-4">
              Pricing
            </Link>
            <Link href="#testimonials" className="text-sm font-medium hover:underline underline-offset-4">
              Testimonials
            </Link>
          </nav>
          <div className="flex items-center gap-4">
            <Link href="/login">
              <Button variant="ghost" size="sm">
                Log in
              </Button>
            </Link>
            <Link href="/signup">
              <Button size="sm">Get Started</Button>
            </Link>
          </div>
        </div>
      </header>

      <main className="flex-1">
        {/* Hero Section */}
        <section className="py-20 md:py-28">
          <div className="container mx-auto max-w-7xl px-4 md:px-6">
            <div className="grid gap-6 lg:grid-cols-2 lg:gap-12 items-center">
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
                    Study Smarter, Not Harder with AI
                  </h1>
                  <p className="max-w-[600px] text-gray-500 md:text-xl dark:text-gray-400">
                    ScribeWise transforms your study materials into personalized learning experiences. Take notes,
                    create flashcards, and ace your exams with AI-powered assistance.
                  </p>
                </div>
                <div className="flex flex-col gap-2 min-[400px]:flex-row">
                  <Link href="/signup">
                    <Button size="lg" className="w-full min-[400px]:w-auto">
                      Start Free Trial
                    </Button>
                  </Link>
                  <Link href="#how-it-works">
                    <Button variant="outline" size="lg" className="w-full min-[400px]:w-auto">
                      Learn More
                    </Button>
                  </Link>
                </div>
              </div>
              <div className="flex justify-center">
                <Image
                  src="/placeholder.svg"
                  alt="ScribeWise Dashboard Preview"
                  width={500}
                  height={500}
                  className="rounded-lg object-cover border shadow-lg"
                  priority
                />
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="py-16 bg-gray-50 dark:bg-gray-900">
          <div className="container mx-auto max-w-7xl px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                  Powerful Features for Effective Learning
                </h2>
                <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                  ScribeWise combines cutting-edge AI with proven study techniques to help you learn faster and retain
                  more.
                </p>
              </div>
            </div>
            <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 mt-12">
              <Card>
                <CardHeader>
                  <BookOpen className="h-10 w-10 text-primary mb-2" />
                  <CardTitle>Smart Notes</CardTitle>
                  <CardDescription>
                    AI-powered note-taking that organizes and highlights key concepts automatically.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Upload your lecture notes or textbook pages and let ScribeWise extract the most important
                    information.
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <Brain className="h-10 w-10 text-primary mb-2" />
                  <CardTitle>Adaptive Flashcards</CardTitle>
                  <CardDescription>Personalized flashcards that adapt to your learning progress.</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Our spaced repetition system ensures you review concepts right before you're about to forget them.
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <Sparkles className="h-10 w-10 text-primary mb-2" />
                  <CardTitle>AI Tutor</CardTitle>
                  <CardDescription>Get instant answers to your questions from your personal AI tutor.</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Stuck on a concept? Ask your AI tutor for explanations, examples, and practice problems.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section id="how-it-works" className="py-16">
          <div className="container mx-auto max-w-7xl px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">How ScribeWise Works</h2>
                <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                  Get started in minutes and transform your study routine.
                </p>
              </div>
            </div>
            <div className="mx-auto grid max-w-5xl grid-cols-1 gap-8 md:grid-cols-3 mt-12">
              <div className="flex flex-col items-center space-y-2 text-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-white">1</div>
                <h3 className="text-xl font-bold">Upload Your Materials</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Upload lecture notes, textbooks, or any study material in various formats.
                </p>
              </div>
              <div className="flex flex-col items-center space-y-2 text-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-white">2</div>
                <h3 className="text-xl font-bold">AI Processing</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Our AI analyzes your content, identifies key concepts, and creates study materials.
                </p>
              </div>
              <div className="flex flex-col items-center space-y-2 text-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-white">3</div>
                <h3 className="text-xl font-bold">Study Smarter</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Review personalized flashcards, take quizzes, and get help from your AI tutor.
                </p>
              </div>
            </div>
            <div className="flex justify-center mt-12">
              <Image
                src="/placeholder.svg"
                alt="ScribeWise Workflow"
                width={800}
                height={400}
                className="rounded-lg object-cover border shadow-lg"
                priority
              />
            </div>
          </div>
        </section>

        {/* Testimonials Section */}
        <section id="testimonials" className="py-16 bg-gray-50 dark:bg-gray-900">
          <div className="container mx-auto max-w-7xl px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">What Our Users Say</h2>
                <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                  Join thousands of students who have transformed their study habits with ScribeWise.
                </p>
              </div>
            </div>
            <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 mt-12">
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <div className="h-10 w-10 rounded-full bg-gray-200 dark:bg-gray-800"></div>
                    <div>
                      <CardTitle className="text-base">Sarah K.</CardTitle>
                      <CardDescription>Medical Student</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    "ScribeWise has been a game-changer for my medical studies. The AI tutor helps me understand complex
                    concepts, and the flashcards make memorization so much easier."
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <div className="h-10 w-10 rounded-full bg-gray-200 dark:bg-gray-800"></div>
                    <div>
                      <CardTitle className="text-base">James T.</CardTitle>
                      <CardDescription>Computer Science Major</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    "I've tried many study apps, but ScribeWise is on another level. The AI actually understands my
                    programming questions and gives helpful explanations."
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <div className="h-10 w-10 rounded-full bg-gray-200 dark:bg-gray-800"></div>
                    <div>
                      <CardTitle className="text-base">Elena M.</CardTitle>
                      <CardDescription>Law Student</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    "ScribeWise helped me organize my case notes and prepare for the bar exam. I'm convinced it saved me
                    hundreds of hours of study time."
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* Pricing Section */}
        <section id="pricing" className="py-16">
          <div className="container mx-auto max-w-7xl px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                  Simple, Transparent Pricing
                </h2>
                <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                  Choose the plan that works best for your study needs.
                </p>
              </div>
            </div>
            <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-3 mt-12">
              <Card className="flex flex-col">
                <CardHeader>
                  <CardTitle>Free</CardTitle>
                  <CardDescription>For casual learners</CardDescription>
                  <div className="mt-4 text-4xl font-bold">$0</div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Forever free</p>
                </CardHeader>
                <CardContent className="flex-1">
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>Basic note organization</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>10 AI-generated flashcards per month</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>Limited AI tutor questions</span>
                    </li>
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" className="w-full">
                    Sign Up Free
                  </Button>
                </CardFooter>
              </Card>
              <Card className="flex flex-col border-primary">
                <CardHeader>
                  <div className="px-4 py-1 text-xs font-bold bg-primary text-primary-foreground rounded-full w-fit mb-2">
                    MOST POPULAR
                  </div>
                  <CardTitle>Student</CardTitle>
                  <CardDescription>For serious students</CardDescription>
                  <div className="mt-4 text-4xl font-bold">$9.99</div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">per month</p>
                </CardHeader>
                <CardContent className="flex-1">
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>Advanced note organization</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>Unlimited AI-generated flashcards</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>100 AI tutor questions per month</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>Study progress tracking</span>
                    </li>
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button className="w-full">Start 7-Day Free Trial</Button>
                </CardFooter>
              </Card>
              <Card className="flex flex-col">
                <CardHeader>
                  <CardTitle>Pro</CardTitle>
                  <CardDescription>For professional learners</CardDescription>
                  <div className="mt-4 text-4xl font-bold">$19.99</div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">per month</p>
                </CardHeader>
                <CardContent className="flex-1">
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>Everything in Student plan</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>Unlimited AI tutor questions</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>Advanced analytics and insights</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="mr-2 h-4 w-4 text-primary" />
                      <span>Priority support</span>
                    </li>
                  </ul>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" className="w-full">
                    Start 7-Day Free Trial
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-16 bg-primary text-primary-foreground">
          <div className="container mx-auto max-w-7xl px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                  Ready to Transform Your Study Habits?
                </h2>
                <p className="max-w-[900px] md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                  Join thousands of students who are studying smarter, not harder.
                </p>
              </div>
              <div className="flex flex-col gap-2 min-[400px]:flex-row">
                <Link href="/signup">
                  <Button size="lg" variant="secondary" className="w-full min-[400px]:w-auto">
                    Start Free Trial
                  </Button>
                </Link>
                <Link href="#features">
                  <Button size="lg" variant="outline" className="w-full min-[400px]:w-auto border-primary-foreground">
                    Learn More
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t py-6 md:py-0">
        <div className="container mx-auto max-w-7xl px-4 flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
          <div className="flex items-center gap-2 font-bold">
            <Sparkles className="h-5 w-5 text-primary" />
            <span>ScribeWise</span>
          </div>
          <div className="flex gap-4 text-sm text-gray-500 dark:text-gray-400">
            <Link href="#" className="hover:underline underline-offset-4">
              Terms
            </Link>
            <Link href="#" className="hover:underline underline-offset-4">
              Privacy
            </Link>
            <Link href="#" className="hover:underline underline-offset-4">
              Contact
            </Link>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400">© 2025 ScribeWise. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}

