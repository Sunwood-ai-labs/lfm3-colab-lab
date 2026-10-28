import { defineConfig } from 'vitepress'

const repository = 'https://github.com/Sunwood-ai-labs/lfm3-colab-lab'

const editLink = {
  pattern: `${repository}/edit/main/docs/:path`,
  text: 'このページを編集',
}

const englishEditLink = {
  pattern: `${repository}/edit/main/docs/:path`,
  text: 'Edit this page',
}

export default defineConfig({
  base: '/lfm3-colab-lab/',
  cleanUrls: true,
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/lfm3-colab-lab/logo.svg' }],
  ],
  locales: {
    root: {
      label: '日本語',
      lang: 'ja-JP',
      title: 'LFM3 Colab Lab',
      description: 'Liquid AI の6モデルを Google Colab の Tesla T4 で試す実験ノートと結果。',
      themeConfig: {
        nav: [
          { text: '概要', link: '/' },
          { text: '実験結果', link: '/results' },
          { text: '実行確認', link: '/verification' },
          { text: 'Python CLI', link: '/python-cli' },
        ],
        sidebar: [
          {
            text: 'ドキュメント',
            items: [
              { text: 'プロジェクト概要', link: '/' },
              { text: '実験結果と限界', link: '/results' },
              { text: '公開ノートブックの実行確認', link: '/verification' },
              { text: 'Python CLI で実行', link: '/python-cli' },
            ],
          },
        ],
        editLink,
        footer: {
          message: '実験コードと説明文は MIT License です。モデルの利用条件は各配布元を確認してください。',
          copyright: 'Copyright © Sunwood AI Labs',
        },
      },
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'LFM3 Colab Lab',
      description: 'Run six Liquid AI models on a Google Colab Tesla T4 and inspect the recorded results.',
      themeConfig: {
        nav: [
          { text: 'Overview', link: '/en/' },
          { text: 'Results', link: '/en/results' },
          { text: 'Run validation', link: '/en/verification' },
          { text: 'Python CLI', link: '/en/python-cli' },
        ],
        sidebar: [
          {
            text: 'Documentation',
            items: [
              { text: 'Project overview', link: '/en/' },
              { text: 'Results and limitations', link: '/en/results' },
              { text: 'Public notebook validation', link: '/en/verification' },
              { text: 'Run the Python CLI', link: '/en/python-cli' },
            ],
          },
        ],
        editLink: englishEditLink,
        footer: {
          message: 'Experiment code and documentation use the MIT License. Check each model source for its terms.',
          copyright: 'Copyright © Sunwood AI Labs',
        },
      },
    },
  },
  themeConfig: {
    logo: {
      src: '/logo.svg',
      alt: 'LFM3 Colab Lab',
    },
    search: {
      provider: 'local',
    },
    socialLinks: [
      { icon: 'github', link: repository },
    ],
  },
})
